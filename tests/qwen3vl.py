import os
import platform
# Disable unsupported CUDA allocator feature on Windows to avoid warnings/crashes
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:false")

import torch
from sympy.physics.units import yocto
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from accelerate import init_empty_weights
from PIL import Image
import argparse
import time
import json
import sys

model_path = "D:\\workspace\\cpp_projects\\ow_nn\\model"

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL inference test")
    parser.add_argument("--image", type=str, nargs='*', default=None, help="Path(s) to image files")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="Text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--quantize", choices=["none", "4bit", "8bit"], default="none")
    # chat template options
    parser.add_argument("--use_chat_template", action="store_true", help="Enable multi-turn chat with images")
    parser.add_argument("--chat_messages", type=str, default=None, help="Path to a JSON list of chat messages")
    parser.add_argument("--system", type=str, default="You are a helpful visual assistant.", help="System prompt for chat mode")
    # interactive options
    parser.add_argument("--interactive", action="store_true", help="Start interactive multi-turn session")
    parser.add_argument("--max_history", type=int, default=20, help="Max stored messages in history")
    return parser.parse_args()

args = parse_args()

def pick_compute_dtype() -> torch.dtype:
    """Pick a safe compute dtype depending on platform and CUDA availability.

    - On Windows, avoid BF16 entirely (known instability/drivers)
    - On CUDA Linux/macOS, prefer BF16 if supported, else FP16
    - On CPU, use FP32
    """
    if torch.cuda.is_available():
        # BF16 causes crashes on some Windows + CUDA stacks; force FP16 there
        if platform.system().lower() == "windows":
            return torch.float16
        # Prefer BF16 when actually supported; else FP16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

# ===== 交互式多轮图文聊天回路 =====

def interactive_chat_loop(processor, model, device, args):
    messages = []
    if args.system:
        messages.append({"role": "system", "content": [{"type": "text", "text": args.system}]})

    image_history = []
    max_history = getattr(args, "max_history", 20)

    def trim_messages():
        nonlocal messages
        if max_history is not None:
            sys_msgs = [m for m in messages if m.get("role") == "system"]
            non_sys = [m for m in messages if m.get("role") != "system"]
            if len(non_sys) > max_history:
                messages = sys_msgs + non_sys[-max_history:]

    # 可选首轮（来自 --prompt / --image）
    initial_images = None
    if args.image:
        try:
            initial_images = [Image.open(p).convert("RGB") for p in args.image]
        except Exception:
            initial_images = None
    if args.prompt or initial_images:
        user_content = []
        if args.prompt:
            user_content.append({"type": "text", "text": args.prompt})
        if initial_images:
            for _ in range(len(initial_images)):
                user_content.append({"type": "image"})
            image_history.extend(initial_images)
        messages.append({"role": "user", "content": user_content})
        trim_messages()
        chat_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=chat_prompt, images=image_history if image_history else None, return_tensors="pt").to(device)
        start = time.time()
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        elapsed = time.time() - start
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("\n--- 推理结果 ---")
        print(output_text)
        print(f"\n耗时: {elapsed:.2f}s | 生成token数: {generated_ids.shape[-1]}")
        messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
        trim_messages()

    print("\n进入交互模式。指令: :img <path...>, :clear, :system <text>, :exit")
    images_buffer = []
    try:
        while True:
            try:
                user_input = input("你: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() == ":exit":
                break
            if user_input.lower().startswith(":clear"):
                messages = []
                image_history = []
                if args.system:
                    messages.append({"role": "system", "content": [{"type": "text", "text": args.system}]})
                print("历史已清空。")
                continue
            if user_input.lower().startswith(":system"):
                sys_prompt = user_input[len(":system"):].strip()
                if messages and messages[0].get("role") == "system":
                    messages[0] = {"role": "system", "content": [{"type": "text", "text": sys_prompt}]}
                else:
                    messages.insert(0, {"role": "system", "content": [{"type": "text", "text": sys_prompt}]})
                print("系统提示已更新。")
                continue
            if user_input.lower().startswith(":img"):
                paths = user_input[len(":img"):].strip().split()
                images_buffer = []
                for p in paths:
                    try:
                        images_buffer.append(Image.open(p).convert("RGB"))
                    except Exception as e:
                        print(f"加载图片失败: {p} -> {e}")
                print(f"已添加 {len(images_buffer)} 张图片到本轮消息。")
                continue

            # 普通文本消息
            user_content = [{"type": "text", "text": user_input}]
            if images_buffer:
                for _ in range(len(images_buffer)):
                    user_content.append({"type": "image"})
                image_history.extend(images_buffer)
                images_buffer = []
            messages.append({"role": "user", "content": user_content})
            trim_messages()

            chat_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=chat_prompt, images=image_history if image_history else None, return_tensors="pt").to(device)
            start = time.time()
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )
            elapsed = time.time() - start
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("助手: ", output_text)
            print(f"(耗时 {elapsed:.2f}s, 生成 {generated_ids.shape[-1]} tokens)")
            messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
            trim_messages()
    except KeyboardInterrupt:
        pass


def format_params(num_params):
    '''Converts a number of parameters into a human-readable string (B, M, K).'''
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f} B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f} M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f} K"
    else:
        return f"{num_params}"

# 1. load model config
print("Loading model configuration...")
config = AutoConfig.from_pretrained(model_path)
print(config)

# 2. use init_empty_weights to manage context (ctx)
# use this context to create model, params of model do not occupy real raw memory
print("create model for  init_empty_weights")

# 可选量化配置，节省显存（Windows/GPU能力自适应）
quantization_config = None
compute_dtype = pick_compute_dtype()
if args.quantize in {"4bit", "8bit"} and not torch.cuda.is_available():
    print("未检测到可用CUDA，量化选项已回退为 none。")
elif args.quantize == "4bit" and torch.cuda.is_available():
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    except Exception as e:
        print(f"4bit量化初始化失败，回退为 none: {e}")
        quantization_config = None
elif args.quantize == "8bit" and torch.cuda.is_available():
    try:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    except Exception as e:
        print(f"8bit量化初始化失败，回退为 none: {e}")
        quantization_config = None

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=compute_dtype,     # 强制安全 dtype，避免 Windows 上 BF16 崩溃
    device_map="auto" if torch.cuda.is_available() else "cpu",
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    trust_remote_code=True,
    use_safetensors=True,
)
# 3. print model struct
print("\n ------------ model struct -----------")
#  check model device
print(f"\n param of model device: {model.device}")
print(f"anyone param of device: {next(model.parameters()).device}")

# --- add a new part for module: print all usage of module's parameters
module_param_counts = []

for name, module in model.named_modules():
    num_params = sum(p.numel() for p in module.parameters())
    if num_params > 0:
        module_param_counts.append((name, num_params))


# order by  result of module name to make readable raise
module_param_counts.sort(key=lambda x:x[0])

print(f"{'module name'}: < 80 {'usage of param'}: > 20")
print(f"{'-'*80} {'-'*20}")

for name, count in module_param_counts:
    print(f"{name:<80} {format_params(count):>20}")

# calculate all model's params summary
total_model_params = sum(p.numel() for p in model.parameters())
print(f"{'-'*80} {'-'*20}")
print(f"{'gross of params':<80} {format_params(total_model_params):>20}")

print("compare major params of module number ---")

vit_params = 0
merger_params = 0
language_model_params = 0
lm_head_params = 0 # lm_head is independent language_model's input layers

for name, count in module_param_counts:
    if name == 'model.visual':
        vit_params = count
    elif name == 'model.visual.merger':
        merger_params += count
    elif name == "model.visual.deepstack_merger_list":
        merger_params += count
    elif name == "model.language_model":
        language_model_params = count
    elif name == "lm_head":
        lm_head_params = count

print(f"{'vit':<30} {format_params(vit_params):>15}")
print(f"{'Merge':<30} {format_params(merger_params):>15}")
print(f"{'not have lm_head':<30} {format_params(language_model_params - lm_head_params):>15}")
print(f"{'hava lm_head':<30} {format_params(lm_head_params):>15}")
print(f"{'sum':<30} {format_params(language_model_params + lm_head_params):>15}")

# 4. 推理：支持图片 + 文本
processor = AutoProcessor.from_pretrained(model_path)
device = next(model.parameters()).device

prompt = args.prompt
image_paths = args.image

images = None
if image_paths:
    try:
        images = [Image.open(p).convert("RGB") for p in image_paths]
    except Exception as e:
        print(f"加载图片失败: {e}")
        images = None

# 交互模式：进入循环并退出主流程
if args.interactive:
    interactive_chat_loop(processor, model, device, args)
    sys.exit(0)

if args.use_chat_template:
    # 构造多轮图文消息
    messages = None
    if args.chat_messages:
        try:
            with open(args.chat_messages, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            messages = loaded.get("messages", loaded) if isinstance(loaded, dict) else loaded
        except Exception as e:
            print(f"加载聊天消息文件失败: {e}，回退到单轮消息构造。")
            messages = None

    if messages is None:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": [{"type": "text", "text": args.system}]})
        user_content = []
        if prompt:
            user_content.append({"type": "text", "text": prompt})
        if images:
            for _ in range(len(images)):
                user_content.append({"type": "image"})
        messages.append({"role": "user", "content": user_content})
    else:
        # 尝试从 messages 中收集 image_url 并加载图片
        collected_images = []
        try:
            for msg in messages:
                for item in msg.get("content", []):
                    if item.get("type") == "image_url":
                        url = item.get("image_url", "")
                        if isinstance(url, str):
                            path = url.replace("file://", "")
                            try:
                                collected_images.append(Image.open(path).convert("RGB"))
                            except Exception:
                                pass
        except Exception:
            pass
        # 将 image_url 统一为 image 类型（数量占位）
        for msg in messages:
            if "content" in msg:
                for i, item in enumerate(msg["content"]):
                    if item.get("type") == "image_url":
                        msg["content"][i] = {"type": "image"}
        if collected_images:
            images = collected_images

    chat_prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=chat_prompt, images=images, return_tensors="pt").to(device)
else:
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)

print("\n开始推理...")
start = time.time()

generated_ids = model.generate(
    **inputs,
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
    do_sample=args.do_sample,
)

elapsed = time.time() - start
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- 推理结果 ---")
print(output_text)
print(f"\n耗时: {elapsed:.2f}s | 生成token数: {generated_ids.shape[-1]}")