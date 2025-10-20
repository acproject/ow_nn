import torch
from sympy.physics.units import yocto
from transformers import AutoConfig, AutoModelForImageTextToText,BitsAndBytesConfig
from accelerate import init_empty_weights

model_path = "D:\\workspace\\cpp_projects\\ow_nn\\model"

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

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype="auto",           # 自动使用 float16/bfloat16
    device_map="auto",            # 根据显存/内存自动分配
    low_cpu_mem_usage=True        # 减少 CPU 端峰值内存
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
    elif name == "model.languae_model":
        language_model_params = count
    elif name == "lm_head":
        lm_head_params = count

print(f"{'vit':<30} {format_params(vit_params):>15}")
print(f"{'Merge':<30} {format_params(merger_params):>15}")
print(f"{'not have lm_head':<30} {format_params(language_model_params - lm_head_params):>15}")
print(f"{'hava lm_head':<30} {format_params(lm_head_params):>15}")
print(f"{'sum':<30} {format_params(language_model_params + lm_head_params):>15}")