## Context

- Context 是一个 CPU 内存管理/分配器，服务于算子执行时的中间与临时缓冲。
- 维护两类内存池：主“线性 Arena”（ arena ，面向较大、可能跨算子的缓冲）和线程局部的“scratch”临时缓冲（TLS，每个线程独立）。
- 提供标记/回滚（LIFO）式释放、对齐分配、可用内存感知的增长策略，以及面向矩阵打包的优化接口。
核心职责

- 主分配器： void* alloc(size_t bytes) 从 arena 线性地切割内存。
- 临时分配器： void* scratch_alloc(size_t bytes) 提供线程安全的短生命周期缓冲。
- 生命周期管理： mark() / release_to(mark) / reset() 用于 LIFO 风格的范围释放。
- 观测指标： used() / capacity() 和 scratch_used() / scratch_capacity() 。
构造与初始状态

- 构造函数： Context(size_t arena_size = 8 * 1024 * 1024, size_t align = 64)
  - 主 arena 初始 8MB； offset = 0 。
  - 线程局部 scratch 在首次使用时初始化为 512KB（TLS），类内保留一个“legacy scratch”成员但已不再使用。
  - align_bytes 记录对齐粒度（用于 scratch 的对齐），默认为 64 字节。
主内存分配策略（alloc）

- 零字节直接返回 nullptr 。
- 对单次请求设置防御：若 bytes > 8GB ，抛出 std::bad_alloc （避免不合理的超大单块）。
- 64 字节对齐（ aligned_bytes = (bytes + 63) & ~63ULL ）。
- 如果现有空间不足：
  - 计算 needed_size = offset + aligned_bytes 。
  - 采用“激进增长”策略：在当前容量基础上，增长量取 max(2GB, 当前容量, 需求缺口)。
  - 读取可用系统内存（ get_available_memory() ），将目标容量限制在“可用内存的 75%”以内；若受限则退为保守增长（至少满足所需）。
  - 扩容采用 arena.resize(new_size) ；失败则尝试“仅满足所需的最小扩容”。
- 成功后返回 arena.data() + offset ，并推进 offset 。
提示：这里使用 std::vector<uint8_t>::resize 实际改变了 size() ，同时可能导致底层重新分配，这会在扩容发生时“使所有先前返回的指针失效”。这是线性 Arena 设计的一个重要注意点（详见“注意事项”）。

可用内存检测（get_available_memory）

- Windows： GlobalMemoryStatusEx 返回物理可用内存。
- 非 Windows：使用 sysconf(_SC_AVPHYS_PAGES) 和 _SC_PAGE_SIZE 估算可用字节数。
- 在 macOS 上， _SC_AVPHYS_PAGES 不存在，会引发编译/静态检查错误（你当前的 linter 报错即来源于此）。
修复建议（macOS）：

- 在 #ifdef __APPLE__ 分支中用 host_statistics64 + vm_statistics64_data_t 统计 free_count 与 inactive_count ，乘以 sysconf(_SC_PAGESIZE) 得到可用字节；或用 sysctl 组合查询。示例思路：
  - 头文件： <mach/mach.h> 与 <sys/sysctl.h>
  - 逻辑： free_bytes = free_count * PAGE_SIZE ， inactive_bytes = inactive_count * PAGE_SIZE ，返回两者之和。
线程局部 scratch（临时缓冲）

- 结构： ThreadScratch { std::vector<uint8_t> buf; size_t offset; } ，通过 thread_local 实现每线程独立。
- 初始化：首次使用时容量 512KB， offset = 0 。
- 分配： scratch_alloc(bytes) 对齐到 align_bytes （默认 64），不足时扩容：
  - 需要量 < 64MB：增长到 max(needed*2, 16MB) （倾向翻倍，最少 16MB）。
  - 需要量 < 512MB：在需要量基础上 +128MB。
  - 超过 512MB：+64MB。
- 专用接口： scratch_alloc_matrix_pack(rows, cols, elem_size) 以 64 字节对齐分配矩阵打包缓冲。
- 复位： scratch_reset() 将该线程的 offset 置零，不收缩容量。
注意：scratch 也使用 std::vector::resize 扩容；一旦扩容发生，之前该线程从 scratch 返回的指针同样会失效。因此，应尽量在一个算子内预估 scratch 总需求并一次性分配，避免“使用旧指针期间触发扩容”。

生命周期辅助（线性/LIFO）

- mark() ：记录当前 offset 。
- release_to(mark) ：若 mark <= offset ，把 offset 回退到该位置（不改变容量）。
- reset() ：把 offset 清零（不改变容量）。
- 适合“范围型生命周期”：在进入一个计算阶段前打标记，用完后整体回退。
观测接口

- used() / capacity() ：主 arena 已用字节与总容量字节（注意这里用的是 arena.size() ）。
- scratch_used() / scratch_capacity() ：当前线程的 scratch 已用与容量。
并发与线程安全

- 主 arena 并不线程安全： offset 是共享状态，多个线程并发调用 alloc 会发生竞态。应：
  - 在多线程场景中每线程使用各自的 Context ，或
  - 用外部同步（mutex）保护 alloc 。
- scratch 是线程局部（TLS），单线程调用无需锁。
典型用法

- 主缓冲：
  - auto ctx = std::make_shared<Context>(/*预估容量*/);
  - size_t m = ctx->mark();
  - void* big = ctx->alloc(N); // 用于较大中间结果
  - /* 计算 */
  - ctx->release_to(m); // 在阶段结束时整体回退
- 临时缓冲（算子内部）：
  - 预估本算子需要的临时总量，尽量一次 scratch_alloc ；
  - 用完后 scratch_reset() （或在下一个算子开始前复位）。
注意事项与改进建议

- 指针失效风险（重要）：
  - 使用 std::vector::resize 扩容会重新分配底层存储，一旦扩容，之前从 arena /scratch 返回的指针全部失效。
  - 规避策略：
    - 尽量在阶段开始前“预估并一次性保证足够容量”，避免在持有旧指针期间触发扩容；
    - 对 scratch，在算子内先估总量一次性分配，或在使用旧指针时避免再次调用会触发扩容的 scratch_alloc 。
    - 若必须保证指针绝对稳定，考虑用一次性大映射（如 mmap 预留）或自定义固定区资源（如 std::pmr::monotonic_buffer_resource 指向固定缓冲）。
- macOS 编译问题：
  - _SC_AVPHYS_PAGES 在 macOS 不存在，需要单独分支实现。见上文修复建议。
- 对齐一致性：
  - alloc 固定以 64 字节对齐，但类构造允许自定义 align_bytes ；建议统一主 alloc 与 scratch 的对齐策略（都使用 align_bytes ），以避免隐式不一致。
- 增长策略保守性：
  - 主 arena 的“至少 2GB 或翻倍”的增长在中小模型场景可能过于激进，易产生内存压力；可考虑按模型规模或配置调整策略。
- 日志打印：
  - alloc 与 scratch 的扩容路径会输出到 std::cout / std::cerr ，在高频路径可能影响性能；建议加开关或调试等级控制。
- 线程安全：
  - 主 arena 不是线程安全；在使用多线程的推理或训练时应采用“每线程一个 Context”或外部同步。
与项目的关系（可能的调用点）

- ops.hpp 、 transformer.cpp 、 tensor.hpp 很可能在算子或张量构造时用到 Context 来管理内存生存期。
- thread_pool.hpp 下的并行算子如使用共享 Context ，需格外注意并发与指针失效问题。
如果你希望，我可以：

- 补上 macOS 的 get_available_memory() 实现分支；
- 将主 alloc 对齐策略改为使用 align_bytes ；
- 为 Context 加一条“预估/预留”接口（例如 reserve_arena(bytes) ），帮助在阶段开始前一次性扩容，降低指针失效风险；
- 给 scratch 增加“预估本算子需求”的助手方法并在日志中标注算子名，便于调优。