# vllm-xpu-kernels Attention Kernel 深度研究报告

## 目录

1. [项目概述](#1-项目概述)
2. [Attention Kernel 总览](#2-attention-kernel-总览)
3. [FlashAttention v2 Kernel 详解](#3-flashattention-v2-kernel-详解)
   - 3.1 [Python 层接口](#31-python-层接口)
   - 3.2 [C++ 层 Flash API](#32-c-层-flash-api)
   - 3.3 [路由选择逻辑：Prefill vs Decode](#33-路由选择逻辑prefill-vs-decode)
   - 3.4 [Chunk Prefill Kernel](#34-chunk-prefill-kernel)
   - 3.5 [Paged Decode Kernel](#35-paged-decode-kernel)
4. [GDN (Gated Delta Network) Attention Kernel 详解](#4-gdn-gated-delta-network-attention-kernel-详解)
   - 4.1 [GDN Attention 工作原理](#41-gdn-attention-工作原理)
   - 4.2 [Causal Conv1d 组件](#42-causal-conv1d-组件)
   - 4.3 [Gated Delta Rule 组件](#43-gated-delta-rule-组件)
   - 4.4 [XE2 加速实现](#44-xe2-加速实现)
5. [完整调用流程](#5-完整调用流程)
   - 5.1 [FlashAttention 调用流程](#51-flashattention-调用流程)
   - 5.2 [GDN Attention 调用流程](#52-gdn-attention-调用流程)
6. [Kernel 选择逻辑与 Policy 分发](#6-kernel-选择逻辑与-policy-分发)
   - 6.1 [Head Size 策略分发](#61-head-size-策略分发)
   - 6.2 [数据类型分发](#62-数据类型分发)
   - 6.3 [Boolean 特性组合分发](#63-boolean-特性组合分发)
   - 6.4 [Decode 策略的额外维度](#64-decode-策略的额外维度)
7. [GPU 硬件适配与架构约束](#7-gpu-硬件适配与架构约束)
8. [支持的特性矩阵](#8-支持的特性矩阵)
9. [关键数据结构与张量布局](#9-关键数据结构与张量布局)
10. [编译与构建体系](#10-编译与构建体系)
11. [测试体系分析](#11-测试体系分析)
12. [重要发现与设计亮点](#12-重要发现与设计亮点)
13. [附录：文件索引](#13-附录文件索引)

---

## 1. 项目概述

`vllm-xpu-kernels` 是一个 vLLM 插件项目，为 Intel GPU（XPU）提供高性能自定义计算 kernel。该项目使用 C++/SYCL 编写 GPU kernel，通过 PyTorch C++ 扩展机制（`torch::library`）暴露给 Python 端，最终集成到 vLLM 推理框架中。

**核心技术栈：**
- 语言：C++/SYCL + Python
- GPU 目标：Intel XPU（PVC / BMG 架构，重点支持 XE2 架构）
- 构建工具链：CMake + setuptools + Intel oneAPI 2025.3
- 外部依赖：
  - **PyTorch 2.10+xpu (libtorch)** — 所有 kernel 共同依赖；提供 `torch::Tensor`、`torch::library` 注册机制、C++ 扩展加载框架
  - **Intel oneAPI DPC++/C++ Compiler 2025.3 运行时** — 所有 kernel 共同依赖；提供 `libsycl.so`（SYCL 运行时）、`libOpenCL.so`（OpenCL 运行时）、SPIR-V translator（离线编译 AOT device code）。CMakeLists 中显式链接 `sycl` 和 `OpenCL`
  - **[Intel sycl-tla](https://github.com/intel/sycl-tla)（CUTLASS 的 SYCL 移植版）** — 通过 CMake `FetchContent` 在构建时拉取；header-only 库（即**不编译为独立的 .so/.a**，所有代码以 C++ 模板头文件形式 `#include` 到本项目的 kernel 源文件中，由本项目的编译器一起编译为最终的 .so）。被 FlashAttention v2（chunk prefill / paged decode）、GDN Attention、Grouped GEMM kernel 使用。sycl-tla 提供的**不是**具体的 attention kernel 实现，而是底层构建块：
    - `cute::gemm()` / `cutlass::gemm::collective::CollectiveMma` — 子组级矩阵乘加原语，attention kernel 内部用它来执行 Q×K 和 Score×V 矩阵乘法
    - `cutlass::fmha::kernel::XeReduceSplitKTileScheduler` 等 — Tile 调度器，负责将 Q/K 序列维度分块映射到 GPU 工作组
    - `cutlass::fmha::collective::CollectiveEpilogue` — Epilogue 抽象，负责将 GEMM 结果写回全局内存（含 softmax rescale）
    - `cutlass::KernelHardwareInfo` / `cutlass::get_sub_group_id()` 等 — 硬件查询与子组索引工具函数
  - **[oneDNN](https://github.com/uxlfoundation/oneDNN)** — 通过 git submodule 引入（`third_party/oneDNN`）；被 `_xpu_C` 模块中的量化 GEMM 操作使用（`csrc/xpu/onednn/`），attention kernel 本身不直接依赖。`_xpu_C` 调用的 oneDNN 算子完整列表：
    - **`dnnl::matmul`**（量化矩阵乘法原语）— 这是唯一使用的 oneDNN 算子类型，通过 `matmul_primitive_create_and_cache()` 创建并缓存。具体有 4 个封装函数对应 4 种量化组合：
      1. `dnnl_matmul_w8a8_fp8()` — FP8×FP8 → FP16/BF16 矩阵乘（`fp8_gemm_w8a8.h`），支持 e5m2 和 e4m3fn 格式，per-tensor/per-token scale
      2. `dnnl_matmul_w8a16_fp8()` — FP16/BF16×FP8 → FP16/BF16 矩阵乘（`fp8_gemm_w8a16.h`），weight-only FP8 量化，per-channel scale
      3. `dnnl_matmul_w4a16_int4()` — FP16/BF16×INT4 → FP16/BF16 矩阵乘（`int4_gemm_w4a16.h`），支持 group quantization + asymmetric zero-point (u4)
      4. `dnnl_matmul_w4a8_int4()` — INT8×INT4 → FP16 矩阵乘（`int4_gemm_w4a8.h`），双重量化 (activation s8/u8 + weight u4)，per-token/per-tensor activation scale
    - 辅助 API：`dnnl::sycl_interop::make_engine()`、`dnnl::sycl_interop::make_stream()`、`dnnl::sycl_interop::make_memory()` — 用于将 PyTorch XPU device/stream/USM 指针桥接到 oneDNN 运行时（`onednn_runtime.h`）

**构建产物（4 个 C++ 扩展模块）及其全部功能 kernel：**

#### 模块 1：`vllm_xpu_kernels._C` — 通用操作（23 个 kernel）

| 分类 | Kernel 函数 | 说明 | 源文件 |
|------|-------------|------|--------|
| **Activation** | `silu_and_mul` | SiLU × gate | `activation.cpp` |
| | `mul_and_silu` | gate × SiLU | `activation.cpp` |
| | `gelu_and_mul` | GELU × gate | `activation.cpp` |
| | `gelu_tanh_and_mul` | GELU-tanh × gate | `activation.cpp` |
| | `gelu_fast` | Fast GELU | `activation.cpp` |
| | `gelu_new` | New GELU | `activation.cpp` |
| | `gelu_quick` | Quick GELU | `activation.cpp` |
| | `swigluoai_and_mul` | SwiGLU(OAI) × gate | `activation.cpp` |
| **LayerNorm** | `rms_norm` | RMS Normalization | `layernorm.cpp` |
| | `fused_add_rms_norm` | 融合 residual Add + RMS Norm | `layernorm.cpp` |
| **位置编码** | `rotary_embedding` | 旋转位置编码 (RoPE) | `pos_encoding_kernels.cpp` |
| **FP8 量化** | `static_scaled_fp8_quant` | 静态 scale FP8 量化 | `quantization/fp8/fp8_quant.cpp` |
| | `dynamic_scaled_fp8_quant` | 动态 per-tensor FP8 量化 | `quantization/fp8/fp8_quant.cpp` |
| | `dynamic_per_token_scaled_fp8_quant` | 动态 per-token FP8 量化 | `quantization/fp8/fp8_quant.cpp` |
| | `per_token_group_fp8_quant` | per-token-group FP8 量化 | `quantization/fp8/fp8_quant.cpp` |
| **KV Cache** | `reshape_and_cache` | KV cache 写入 | `cache.cpp` |
| | `reshape_and_cache_flash` | KV cache 写入（Flash 布局） | `cache.cpp` |
| | `concat_and_cache_mla` | MLA KV cache 拼接写入 | `cache.cpp` |
| | `gather_cache` | cache block 聚集读取 | `cache.cpp` |
| | `convert_fp8` | FP8 ↔ FP16/BF16/FP32 格式转换 | `cache.cpp` |
| | `swap_blocks` | cache block 换入/换出 | `cache.cpp` |
| **工具** | `weak_ref_tensor` | 弱引用 tensor | `tensor_utils.cpp` |
| | `get_xpu_view_from_cpu_tensor` | CPU tensor → XPU view 映射 | `xpu_view.cpp` |

#### 模块 2：`vllm_xpu_kernels._vllm_fa2_C` — FlashAttention v2（1 个 kernel）

| Kernel 函数 | 说明 | 源文件 |
|-------------|------|--------|
| `varlen_fwd` | FlashAttention v2 variable-length forward（统一入口，内部分发 chunk prefill / paged decode） | `flash_attn/flash_api.cpp` → `xpu/attn/attn_interface.cpp` |

#### 模块 3：`vllm_xpu_kernels._moe_C` — Mixture of Experts（10 个 kernel）

| Kernel 函数 | 说明 | 源文件 |
|-------------|------|--------|
| `moe_sum` | 各专家部分结果求和 | `moe/moe_align_sum_kernels.cpp` |
| `moe_align_block_size` | token → expert 对齐（按 block_size 整除） | `moe/moe_align_sum_kernels.cpp` |
| `batched_moe_align_block_size` | batched 版对齐 | `moe/moe_align_sum_kernels.cpp` |
| `moe_lora_align_block_size` | MoE + LoRA 联合对齐 | `moe/moe_align_sum_kernels.cpp` |
| `grouped_topk` | 分组 TopK 路由选择 | `moe/grouped_topk.cpp` |
| `fused_grouped_topk` | 融合分组 TopK 路由 | `moe/fused_grouped_topk.cpp` |
| `topk_softmax` | TopK + Softmax 门控 | `moe/topk.cpp` |
| `topk_sigmoid` | TopK + Sigmoid 门控 | `moe/topk.cpp` |
| `moe_gather` | MoE 输出聚集（加权合并） | `moe/moe_gather.cpp` |
| `fused_moe_prologue` | MoE 前序：输入量化 + token 排列 | `moe/fused_moe_prologue.cpp` |

#### 模块 4：`vllm_xpu_kernels._xpu_C` — XPU 专用操作（12 个 kernel）

| 分类 | Kernel 函数 | 说明 | 源文件 | 依赖 |
|------|-------------|------|--------|------|
| **量化 GEMM** | `fp8_gemm` | FP8×FP8 → FP16/BF16 矩阵乘 | `xpu/onednn/onednn_matmul.cpp` | oneDNN |
| | `fp8_gemm_w8a16` | FP16/BF16×FP8 weight-only 矩阵乘 | `xpu/onednn/onednn_matmul.cpp` | oneDNN |
| | `int4_gemm_w4a16` | FP16/BF16×INT4 矩阵乘（group quant） | `xpu/onednn/onednn_matmul.cpp` | oneDNN |
| | `int4_gemm_w4a8` | INT8×INT4 双重量化矩阵乘 | `xpu/onednn/onednn_matmul.cpp` | oneDNN |
| **Grouped GEMM** | `cutlass_grouped_gemm_interface` | MoE Grouped GEMM（支持 FP16/BF16/INT4/MXFP4） | `xpu/grouped_gemm/` | sycl-tla |
| **位置编码** | `deepseek_scaling_rope` | DeepSeek 缩放旋转位置编码 | `xpu/sycl/deepseek_scaling_rope.cpp` | — |
| **LoRA** | `bgmv_shrink` | LoRA 低秩投影（A 矩阵，shrink） | `xpu/lora/lora_shrink.cpp` | — |
| | `bgmv_expand` | LoRA 扩展投影（B 矩阵，expand） | `xpu/lora/lora_expand.cpp` | — |
| | `bgmv_expand_slice` | LoRA 切片扩展投影（QKV 拼接场景） | `xpu/lora/lora_expand.cpp` | — |
| **Attention** | `gdn_attention` | Gated Delta Network 注意力 | `xpu/gdn_attn/gdn_attn_interface.cpp` | sycl-tla |
| **设备查询** | `is_bmg` | BMG (Battlemage) 架构检测 | `xpu/utils.cpp` | — |
| | `is_pvc` | PVC (Ponte Vecchio) 架构检测 | `xpu/utils.cpp` | — |

> **合计：4 个模块，46 个注册 kernel 函数。**

---

## 2. Attention Kernel 总览

本项目实现了 **两大类** Attention Kernel，分别服务于不同的模型架构：

### 2.1 FlashAttention v2（标准 Transformer Attention）

用于标准 Transformer 架构中的 Multi-Head Attention / Grouped-Query Attention (MHA/GQA)，是 vLLM 推理的主力 attention kernel。

**包含两个子 kernel：**

| Kernel | 用途 | 触发条件 |
|--------|------|----------|
| **Chunk Prefill** | 处理 prefill 阶段（长序列多 token 输入） | `max_seqlen_q > 1` 或非 paged 模式 |
| **Paged Decode** | 处理 decode 阶段（单 token 自回归生成） | `max_seqlen_q == 1` 且 paged 模式 |

#### Chunk Prefill 工作原理

Chunk Prefill 用于 **prefill 阶段**：用户输入一整段 prompt（多个 token），需要一次性计算所有 token 对之间的 attention。核心算法是 **分块（tiled）FlashAttention v2**，在 Intel XE2 GPU 上使用 CUTLASS SYCL（sycl-tla）的 GEMM 原语实现。

**算法步骤：**

```
输入: Q [seq_q, head_dim], K [seq_k, head_dim], V [seq_k, head_dim]
输出: O [seq_q, head_dim]

1. 分块: 将 Q 按 seq_q 维度切成大小为 TileM 的块（如 128），
         将 K/V 按 seq_k 维度切成大小为 TileN 的块（如 32/64）

2. 对于每个 Q 块（Q_tile, 由一个 work group 负责）:
   初始化: O_acc = 0, max_logit = -∞, sum_exp = 0

   3. 对于每个 K 块（循环遍历 K_tile[0..K_blocks-1]）:
      a) GEMM-1: S_tile = Q_tile × K_tile^T           (使用 cute::gemm + XE_DPAS 指令)
         → 得到 attention score 子矩阵 [TileM, TileN]

      b) Masking: 应用 causal mask / sliding window mask / 边界 mask
         → 将无效位置设为 -∞

      c) Online Softmax（分块安全 softmax）:
         - new_max = max(old_max, scale * row_max(S_tile))
         - rescale = exp2(old_max - new_max)
         - S_tile = exp2(scale * S_tile - new_max)      (逐元素)
         - sum_exp = sum_exp * rescale + row_sum(S_tile)
         - O_acc = O_acc * rescale                       (修正之前累积的输出)

      d) GEMM-2: O_acc += S_tile × V_tile               (使用 cute::gemm + XE_DPAS 指令)
         → 累积 attention 加权的 value

   4. Epilogue: O = O_acc / sum_exp                      (归一化)
      → 写回全局内存
```

**关键设计要点：**
- **Online Softmax**：不需要两遍扫描（先求全局 max/sum，再归一化），而是在遍历 K 块的过程中增量更新 max 和 sum，并通过 `rescale` 因子修正之前累积的 O 值。这使得整个计算只需一遍扫描。
- **Tiling 策略**：`TileShapeQK`（如 128×64×32）的三个维度分别对应 Q 方向、K 方向、head_dim 方向的切分。head_dim 方向如果 > 32 则需要多次迭代（内循环 D）。
- **双流水线（2 Pipeline Stages）**：在计算当前 K 块的 GEMM-1 时，同时预取下一个 K 块的数据（`prefetch_k`），隐藏内存延迟。
- **Paged KV Cache 支持**：当 `is_paged=true` 时，K/V 不在连续内存中。通过 `block_table`（页表）将逻辑 K 块索引映射到物理页地址（`get_paged_idx()`），对 kernel 内部透明地处理内存间接寻址。
- **GQA 支持**：多个 Q head 共享同一组 KV head，通过 `head_group_q = num_heads_q / num_heads_kv` 计算分组系数。
- **Variable-Length Batching**：通过 `cu_seqlens_q/cu_seqlens_k`（cumulative sequence lengths）数组支持 batch 内不同长度的序列，每个序列的起始偏移动态计算。

**GPU 执行模型：**
```
Grid: (V_tiles, Q_tiles, batch × num_heads_q)
Block: SGPerWG × 16 threads（每个 subgroup 16 线程，共 SGPerWG 个 subgroup）

每个 Work Group:
  → 负责一个 (Q_tile, V_tile_slice, head, batch) 组合
  → 内部遍历所有 K 块
  → 使用 Intel XE DPAS（Dot Product Accumulate Systolic）指令做矩阵乘法
  → 通过 Block 2D Copy 指令高效加载 Q/K/V 数据
```

#### Paged Decode 工作原理

Paged Decode 用于 **decode 阶段**：自回归生成时每步只产生一个新 token，需要用这个单 token 的 query 对整个 KV cache 做 attention。由于 Q 只有 1 个 token 而 KV 可能很长（数千甚至数万），核心挑战是如何充分利用 GPU 并行度。

**核心策略：Split-KV 并行**

不同于 prefill 可以沿 Q 维度并行（多 token），decode 的 Q 维度只有 1，无法并行。因此采用 **Split-KV** 策略：将长 KV 序列拆成多个分片（splits），每个分片由独立的 work group 并行计算，最后通过一个 reduction kernel 合并结果。

**算法步骤（Phase 1 — Split-KV Attention）：**

```
输入: Q [1, head_dim], K_cache [seq_k, head_dim] (paged), V_cache [seq_k, head_dim] (paged)
参数: num_kv_splits = ceil(num_xe_cores / (batch × num_heads_kv))

1. 将 KV 序列均匀分成 num_kv_splits 个分片:
   split_i 负责 K 块 [start_i, end_i)

2. 对于每个 split_i（由一个 work group 负责）:
   初始化: O_acc = 0, max_logit = -∞, sum_exp = 0

   3. 对于该分片内的每个 K 块:
      a) GEMM-1: S = Q × K_block^T
      b) Masking: 边界处理（sliding window 等）
      c) Online Softmax: 更新 max_logit, sum_exp, O_acc（同 prefill）
      d) GEMM-2: O_acc += S × V_block

   4. Epilogue: 将 O_acc / sum_exp、sum_exp、max_logit 写入临时缓冲区
      → tmp_out[split_i], exp_sums[split_i], max_logits[split_i]
```

**算法步骤（Phase 2 — ReduceSplitK）：**

```
输入: tmp_out[0..S-1], exp_sums[0..S-1], max_logits[0..S-1]（S = num_kv_splits）
输出: O [1, head_dim]

1. 计算全局 max:
   global_max = max(max_logits[0], ..., max_logits[S-1])

2. 加权合并:
   O_final = 0, total_exp = 0
   for i in 0..S-1:
     rescale = exp2(max_logits[i] - global_max)
     O_final += tmp_out[i] × exp_sums[i] × rescale     // tmp_out 在 Phase 1 epilogue 中已归一化（÷ local_exp_sum），此处乘回 exp_sums[i] 恢复未归一化值再参与全局合并
     total_exp += exp_sums[i] × rescale

3. 归一化: O = O_final / total_exp
```

**关键设计要点：**
- **num_kv_splits 自动调节**：`get_num_splits()` 根据 GPU 的 XE Core 数量（`slices × subslices_per_slice`）和 `batch × num_heads_kv` 的并行度自动计算，目标是让所有 XE Core 都有工作：`num_splits = ceil(xe_cores / (batch × heads_kv))`，同时受 `max_seqlen_k / block_size` 上限约束。
- **GQA Packing**：在 decode 中，同一 KV head 对应的多个 Q head（head_group_q 个）被打包在同一个 work group 内处理。Q 的 shape 变为 `[head_group_q, head_dim]`，一次 GEMM 同时计算多个 Q head 的 attention score。这避免了为每个 Q head 重复加载 K/V 数据。
- **Causal Mask 强制关闭**：decode 阶段 query 位于位置 t，KV cache 仅包含位置 0 到 t-1 的历史 token（由 `seqused_k` 限定有效长度），不存在位置 ≥ t 的"未来"token，因此 causal masking 是多余的。代码中显式传入 `is_causal=false`，因为如果错误地启用会触发一个 seq_len 偏移公式，导致读取无效 KV cache 条目。
- **单流水线（1 Pipeline Stage）**：decode 的每个分片处理的 K 块数相对较少，双流水线收益不大，使用 1 stage 减少寄存器压力。
- **Paged KV Cache**：decode 必须使用 paged KV cache（KV 数据分布在不连续的物理页上）。每个 K 块通过页表（`block_table`）查找实际物理地址。
- **两阶段执行**：Phase 1（Split-KV Attention）和 Phase 2（ReduceSplitK）是两个独立的 SYCL kernel，通过 `EventManager` 管理的 SYCL event 隐式同步。当 `num_kv_splits == 1` 时跳过 Phase 2，直接输出。

**GPU 执行模型：**
```
Phase 1 Grid: (V_tiles, Q_tiles, batch × num_heads_kv × num_kv_splits)
  → Z 维度按 (batch, head_kv, split_idx) 三维展开
  → 每个 Work Group: 负责一个 (batch, kv_head, split) 组合
  → 内部展开 head_group_q 个 Q head

Phase 2 Grid: (1, seq_len_qo, batch × num_heads_q)
  → 每个 Work Group: 负责一个 (batch, q_head, seq_pos) 组合
  → 遍历所有 splits 做 reduce
```

### 2.2 GDN Attention（Gated Delta Network Attention）

用于 Mamba/SSM 类模型架构（如 Qwen-Next），是一种线性复杂度的 attention 替代方案。

**包含两个子组件：**

| 组件 | 功能 |
|------|------|
| **Causal Conv1d** | 因果卷积预处理（提取时序特征） |
| **Gated Delta Rule** | 门控增量规则 SSM 状态更新 |

---

## 3. FlashAttention v2 Kernel 详解

### 3.1 Python 层接口

**入口文件：** `vllm_xpu_kernels/flash_attn_interface.py`

```python
def flash_attn_varlen_func(
    q, k, v,
    max_seqlen_q, cu_seqlens_q,
    max_seqlen_k, cu_seqlens_k=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    window_size=None,      # 滑动窗口 (left, right)
    softcap=0.0,
    block_table=None,      # paged KV cache 块表
    s_aux=None,            # softmax sink
    k_descale=None,        # FP8 KV 反量化因子
    v_descale=None,
    num_splits_kv=None,    # KV 块分割数
    fa_version=2,          # FA 版本选择
    ...
)
```

**关键校验逻辑：**
- `cu_seqlens_k` 和 `seqused_k` 互斥：前者用于非 paged prefill，后者用于 paged 模式
- `block_table` 存在时必须提供 `seqused_k`
- `k_descale` 和 `v_descale` 必须同时提供或同时为 None
- `softmax_scale` 默认为 `head_size^(-0.5)`

**底层调用：**
```python
torch.ops._vllm_fa2_C.varlen_fwd(q, k, v, out, cu_seqlens_q, ...)
```

### 3.2 C++ 层 Flash API

**入口文件：** `csrc/flash_attn/flash_api.cpp`

`mha_varlen_fwd` 函数是 C++ 层的核心入口，注册为 PyTorch 自定义算子 `_vllm_fa2_C.varlen_fwd`。

**类型适配机制（pytorch_shim.h）：**

由于 FlashAttention 原始 C++ API 使用的类型（`int`, `float`, `std::optional<T>&`）与 PyTorch library 绑定不兼容，项目使用了一个精巧的 **类型垫片（shim）** 机制：

```cpp
// 类型映射：
// int        -> int64_t
// float      -> double
// std::optional<T>&        -> const std::optional<T>&
// std::optional<const T>&  -> const std::optional<T>&
```

通过 `make_pytorch_shim` 模板函数自动完成类型转换，使得对上游 FlashAttention 代码的修改最小化。

### 3.3 路由选择逻辑：Prefill vs Decode

在 `flash_api.cpp` 的 `mha_varlen_fwd` 中，根据 `max_seqlen_q` 和 `is_paged` 的组合决定走哪条路径：

```
if (max_seqlen_q > 1 || !is_paged):
    → cutlass_chunk_prefill_interface()    # Prefill 路径
else:  # max_seqlen_q == 1 && is_paged
    → cutlass_paged_decode_interface()     # Decode 路径
```

**Decode 路径的重要优化逻辑：**

1. **num_splits 自动计算**（`get_num_splits` 函数）：
   - 获取 GPU 的 XE Core 数量（slices × subslices per slice）
   - 计算并行度 = `batch_size * num_heads_kv`
   - 目标：确保每个 XE Core 都有工作分配
   - `num_splits = ceil(num_xe_cores / (batch_size * num_heads_kv))`
   - 受 max_seqlen_k / block_size 上限约束

2. **临时存储分配**：
   - `tmp_out`：中间输出，shape `[num_tokens, num_heads_q * num_kv_splits, head_dim]`
   - `max_logits`：每个 split 的最大 logit 值
   - `exp_sums`：每个 split 的指数和

3. **Causal 掩码在 decode 中的特殊处理**：
   ```cpp
   // 对于 paged decode（每个序列单个 query），causal masking 是一个 no-op：
   // seqused_k 已经将 KV 限制为只有有效的过去 token，不存在"未来" token 需要 mask。
   // 传入 is_causal=true 会触发一个 seq_len 公式，添加额外的 KV 位置，
   // 导致无效的缓存条目污染 attention 输出。
   cutlass_paged_decode_interface(..., false /* is_causal: always false for decode */);
   ```

4. **滑动窗口的有效化**：
   ```cpp
   int eff_window_left = window_size_left == -1 ? max_seqlen_k : window_size_left;
   int effective_seqlen_k = is_local ? min(max_seqlen_k, eff_window_left + 1) : max_seqlen_k;
   ```

### 3.4 Chunk Prefill Kernel

#### 3.4.1 架构分层

```
cutlass_chunk_prefill_interface()          // attn_interface.cpp - 架构分发层
  └─ cutlass_chunk_prefill_xe2()           // fmha_xe2.cpp - XE2 适配层
       └─ cutlass_chunk_prefill_impl()     // fmha_xe2.cpp - 核心实现
            └─ policy_dispatch_func<>()    // chunk_prefill_utils.hpp - 策略分发
                 └─ policy_dispatch_impl<>()  // chunk_prefill.hpp - 模板实例化
                      └─ FMHAConfig::kernel_dispatch()  // 最终 kernel 配置
                           └─ FMHAConfig::run<Scheduler>()
                                └─ KernelLauncher::run()  // SYCL 提交
```

#### 3.4.2 Tile Policy 配置

每种 head size 有独立的 tile 策略，定义了 QK GEMM、PV GEMM 和输出的 tile shape：

| Policy | Head Size | ShapeQK | ShapePV | ShapeOut | SubgroupLayoutQK |
|--------|-----------|---------|---------|----------|------------------|
| `chunk_policy_head64` | ≤64 | 128×32×32 | 128×32×32 | 128×64 | 8×1×1 |
| `chunk_policy_head96` | ≤96 | 128×64×32 | 128×32×64 | 128×96 | 8×1×1 |
| `chunk_policy_head128` | ≤128 | 128×64×32 | 128×32×64 | 128×128 | 16×1×1 |
| `chunk_policy_head192` | ≤192 | 256×32×32 | 256×32×32 | 256×192 | 32×1×1 |
| `chunk_policy_head256` | ≤256 | 256×32×32 | 256×32×32 | 256×256 | 32×1×1 |

**设计思路：**
- ShapeQK 的 M 维度（第一维）对应 Q 的序列长度方向的 tile 大小
- ShapeQK 的 N 维度（第二维）对应 K 的序列长度方向的 tile 大小
- ShapeQK 的 K 维度（第三维）对应 head_dim 方向的 tile 大小
- 更大的 head size 需要更大的 work group（更多的 subgroup）

#### 3.4.3 核心计算流程（XeFMHAFwdKernel::operator()）

```
1. 获取当前 tile 的坐标 (blk_q, blk_v, head_q, idx_b)
2. 计算当前序列的 Q/K 长度（支持 variable length）
3. 计算 causal mask 和 local window 的 K 块范围
4. 计算 varlen 模式下的偏移量
5. 构建 Q, K, V, O 的 Tensor 视图
6. 调用 CollectiveMainloop 执行 Q*K + Softmax + P*V
7. 同步 shared memory
8. 调用 CollectiveEpilogue 写回输出（含 softmax sink 处理）
```

#### 3.4.4 Tile Scheduler

使用 `XeFHMAIndividualTileScheduler`，每个 work group 处理一个固定的 tile：

```
Grid(V_tiles, Q_tiles, batch * num_heads_q)
```

其中：
- X 维度：head_dim 方向的 V tile
- Y 维度：Q 序列方向的 tile
- Z 维度：batch × heads（通过 FastDivmod 拆分）

### 3.5 Paged Decode Kernel

#### 3.5.1 与 Prefill 的关键区别

| 特性 | Chunk Prefill | Paged Decode |
|------|---------------|--------------|
| Q 序列长度 | 多 token（>1） | 单 token（==1） |
| KV 来源 | 连续内存或 paged | 必须 paged |
| 并行策略 | 按 Q tile 并行 | KV split 并行 |
| Pipeline Stages | 2 | 1 |
| 输出 | 直接写出 | 需要 reduce |
| Causal Mask | 可选 | 强制关闭 |

#### 3.5.2 Decode Policy 配置

Decode kernel 有三个维度的策略参数化：

1. **Q Group Size**（`_8` 或 `_16`）：根据 `num_heads_q / num_heads_kv` 决定
2. **Head Dimension**（`_64` ~ `_256`）：5 种 head size
3. **Page Size**（`_64` 或 `_128`）：KV cache 的块大小

命名格式：`decode_policy_q{group}_h{head}_p{page}`

例如 `decode_policy_q8_h128_p64` 表示 Q 组大小 8、head 128、page 大小 64。

共 20 种策略组合 = 2(Q Group) × 5(Head Size) × 2(Page Size)

#### 3.5.3 Split-KV 机制

Paged Decode 使用 Split-KV 策略将长 KV 序列拆分成多个分片，各分片独立计算，最后通过 reduction 合并：

```
Step 1: XeFMHAFwdSplitKVKernel
  - 每个 split 计算部分 QK^T softmax PV
  - 输出 partial output、exp_sums、max_logits

Step 2: ReduceSplitK
  - 合并所有 splits 的结果
  - 使用 online softmax 公式修正：
    new_output = (old_exp_sum * old_output + new_exp_sum * new_output) / total_exp_sum
```

Grid 配置：
```
grid.z = batch * num_heads_kv * num_kv_splits  // 注意是 kv_heads，不是 q_heads
```

这意味着对于 GQA，同一 KV head 的多个 Q heads 在同一 work group 内处理（通过 head_group_q 展开）。

#### 3.5.4 DecodeKernelLauncher::run

Decode kernel 的启动包含两个阶段：

```cpp
// Phase 1: Split-KV attention computation
auto event = launch<cutlass::device_kernel<FMHAKernel>>(policy, queue, params);

// Phase 2: Reduction (only when num_kv_splits > 1)
if (need_reduce) {
    auto reduce_event = launch<cutlass::device_kernel<ReductionSplitKernel>>(
        reduce_policy, queue, reduce_params);
}
```

两个 kernel 之间通过 SYCL event 隐式同步（EventManager 管理）。

---

## 4. GDN (Gated Delta Network) Attention Kernel 详解

### 4.1 GDN Attention 工作原理

GDN Attention 是一种 **线性复杂度** 的 attention 替代机制，源自 Gated Delta Networks 论文。它的核心思想是维护一个可学习的 **SSM 状态矩阵**，通过门控的增量更新规则实现类似 attention 的信息聚合。

**数学公式：**

```
O(t) = S(t) · q(t)                                           # 输出 = 状态 × 查询
S(t) = g(t) · S(t-1) + (v(t) - g(t)·S(t-1)·k(t)) · β(t) · k(t)  # 状态更新

其中：
  β(t) = sigmoid(b(t))                        # beta 门控
  g(t) = exp(-exp(A_log) · softplus(a(t) + dt_bias))  # 衰减门控
  q(t), k(t) 经过 L2 归一化
```

**与标准 Attention 的对比：**

| 特性 | Standard Attention | GDN Attention |
|------|-------------------|---------------|
| 复杂度 | O(n²) | O(n)（顺序依赖） |
| 并行性 | 高度并行 | Token 间有依赖 |
| 内存 | 需要完整 KV cache | 固定大小 SSM state |
| 适用场景 | Transformer | Mamba/SSM 类模型（如 Qwen-Next） |

### 4.2 Causal Conv1d 组件

**文件：** `csrc/xpu/gdn_attn/causal_conv1d.hpp`

Causal Conv1d 是 GDN Attention 的前置处理步骤，对混合投影结果（Q, K, V, Z）执行因果一维卷积，提取局部时序特征。

**处理流程：**

```
Input: projected_states_qkvz → Split → [Q, K, V, Z]
                                         ↓
       projected_states_ba   → Split → [B, A]

QKV = concat(Q, K, V)  → Causal Conv1d → [Q_out, K_out, V_out]
Z                       → 直接传出

Conv1d 使用:
  - conv_weights: [mixed_qkv_size, width]
  - conv_bias: [mixed_qkv_size] (可选)
  - conv_state: [batch, width-1, mixed_qkv_size] (保存历史状态)
```

**Kernel 实现细节：**
- Subgroup size = 32，Group size = 256
- 每个 work item 处理 4 个元素（向量化）
- 支持 `silu` 和 `swish` 激活函数
- Conv state 在推理过程中持续更新（支持增量推理）

### 4.3 Gated Delta Rule 组件

**文件：** `csrc/xpu/gdn_attn/gated_delta_rule.hpp`

这是 GDN Attention 的核心计算组件，实现 SSM 状态的门控增量更新。

**Kernel 配置：**
```cpp
group_size = 256;           // 每个 work group 256 个 work item
sg_per_group = 8;           // 8 个 subgroup
v_dim_per_sg = 4;           // 每个 subgroup 处理 4 个 V 维度
k_bucket_size = head_k_dim / 32;  // K 维度的分桶大小
```

**并行维度：**
```
Grid: (batch_size, num_v_heads, num_v_buckets)
Local: (1, 1, 256)
```

**计算流程（每个 token 顺序执行）：**

```
for each token t:
  1. 计算 β(t) = sigmoid(b(t)), g(t) = exp(A_log_exp * softplus(a(t) + dt_bias))
  2. 加载 q(t), k(t) 并进行 L2 归一化
  3. S(t) *= g(t)                    // 衰减旧状态
  4. kv_mem = S(t) · k(t)            // 状态-key 乘积（需要 subgroup reduce）
  5. delta = (v(t) - kv_mem) * β(t)  // 增量
  6. S(t) += k(t) ⊗ delta            // 外积更新
  7. O(t) = S(t) · q(t)              // 输出（需要 subgroup reduce）
```

**数据类型支持：** `bfloat16`, `half`, `float`

**k_bucket_size 分发：** 支持 1, 2, 4, 8（对应 head_k_dim = 32, 64, 128, 256）

### 4.4 XE2 加速实现

**目录：** `csrc/xpu/gdn_attn/xe_2/`

XE2 架构提供了 chunk-based 的 GDN 加速实现：

| 文件 | 功能 |
|------|------|
| `chunk_causal_conv1d_xe2.hpp` | XE2 优化的分块因果卷积 |
| `chunk_gated_delta_rule_xe2.cpp` | XE2 优化的分块门控增量规则 |
| `chunk_gated_delta_rule_kernels_xe2.hpp` | XE2 GEMM kernel |
| `gemm.hpp` | 底层 GEMM 操作 |

**XE2 加速的核心改进：**
- **分块处理**：使用 chunk_size = 64，将长序列分块处理
- **额外填充**：为每个 batch 添加 `chunk_size - 1` 个 padding token
- **矩阵运算加速**：利用 XE2 的 DPAS（Dot Product Accumulate Systolic）指令

**选择逻辑：**
```cpp
if (num_prefills > 0) {
    // XE2 chunk-based 路径（高效）
    chunk_causal_conv1d_xe2(...);
    chunk_gated_delta_rule_xe2(...);
} else {
    // 通用 SYCL 路径
    causal_conv1d(...);
    gated_delta_rule(...);
}
```

注意：只有 **prefill 模式** 使用 XE2 加速路径，decode 模式仍使用通用 SYCL kernel（因为 decode 只处理单 token，chunk 化无收益）。

---

## 5. 完整调用流程

### 5.1 FlashAttention 调用流程

```
Python: flash_attn_varlen_func()
  │  [vllm_xpu_kernels/flash_attn_interface.py]
  │
  │  参数验证 + 类型检查
  │  构建 window_size tuple
  │  处理 dummy cu_seqlens_k
  │
  ▼
C++ Op: torch.ops._vllm_fa2_C.varlen_fwd()
  │  [csrc/flash_attn/flash_api.cpp → TORCH_LIBRARY_EXPAND]
  │
  │  pytorch_shim 类型转换 (int64→int, double→float 等)
  │
  ▼
mha_varlen_fwd()
  │  [csrc/flash_attn/flash_api.cpp]
  │
  │  dtype 验证 (fp16/bf16)
  │  设备验证, 连续性验证
  │  block_table 验证
  │
  ├── if max_seqlen_q > 1 || !is_paged  ─────────────────────┐
  │                                                           │
  │   cutlass_chunk_prefill_interface()                       │
  │     │  [csrc/xpu/attn/attn_interface.cpp]                │
  │     │                                                     │
  │     │  is_xe2_arch() 检查                                 │
  │     │                                                     │
  │     ▼                                                     │
  │   cutlass_chunk_prefill_xe2()                             │
  │     │  [csrc/xpu/attn/xe_2/fmha_xe2.cpp]                 │
  │     │                                                     │
  │     ▼                                                     │
  │   cutlass_chunk_prefill_impl()                            │
  │     │  解析 tensor 维度                                    │
  │     │  处理 window_size / causal 交互                      │
  │     │  检测 FP8 KV cache                                   │
  │     │  构建 chunk_prefill_args_t                           │
  │     │                                                     │
  │     │  根据 head_size 选择 policy:                         │
  │     │  ≤64 → chunk_policy_head64                          │
  │     │  ≤96 → chunk_policy_head96                          │
  │     │  ≤128 → chunk_policy_head128                        │
  │     │  ≤192 → chunk_policy_head192                        │
  │     │  ≤256 → chunk_policy_head256                        │
  │     │                                                     │
  │     ▼                                                     │
  │   policy_dispatch_func<policy>()                          │
  │     │  递归展开 bool 参数 (paged, causal, local, sink)     │
  │     ▼                                                     │
  │   policy_dispatch_impl<policy, Paged, Causal, Local, Sink>│
  │     │  dtype 分发 (half, bf16, fp8_e4m3, fp8_e5m2)        │
  │     ▼                                                     │
  │   FMHAConfig<...>::kernel_dispatch()                      │
  │     │  构建 CUTLASS kernel 类型                            │
  │     ▼                                                     │
  │   FMHAConfig<...>::run<XeFHMAIndividualTileScheduler>()   │
  │     │  构建 SYCL Kernel                                    │
  │     ▼                                                     │
  │   KernelLauncher::run()                                   │
  │     │  SYCL nd-range 提交                                  │
  │     ▼                                                     │
  │   XeFMHAFwdKernel::operator()  ← GPU 上执行                │
  │     1. 获取 tile 坐标 (blk_q, blk_v, head, batch)         │
  │     2. 计算 variable length 的序列长度                      │
  │     3. 应用 causal/local mask 计算 K 块范围                 │
  │     4. 构建 Q, K, V 的 CUTLASS Tensor                      │
  │     5. CollectiveMainloop: QK^T → softmax → PV             │
  │     6. CollectiveEpilogue: 写回 O（含 softmax sink）        │
  │                                                           │
  └── else (max_seqlen_q == 1 && is_paged) ──────────────────┘
      │
      │  计算 effective window size
      │  计算 num_kv_splits (get_num_splits)
      │  分配 tmp_out, max_logits, exp_sums
      │
      ▼
    cutlass_paged_decode_interface()
      │  [csrc/xpu/attn/attn_interface.cpp]
      │
      ▼
    cutlass_paged_decode_xe2()
      │  [csrc/xpu/attn/xe_2/paged_decode_xe2.cpp]
      │
      ▼
    cutlass_paged_decode_impl()
      │  FP8 验证, 维度解析
      │  构建 paged_decode_args_t
      │
      │  dispatch_by_page_size<QGroup>(page_size, ...)
      │    └─ dispatch_by_head_size<QGroup, PageSize>(head_case, ...)
      │         └─ decode_policy_dispatch_func<Policy>(...)
      │              └─ decode_policy_dispatch_impl<Policy, Causal, Local, Sink>(...)
      │                   └─ PagedDecodeConfig<...>::kernel_dispatch()
      │                        └─ DecodeKernelLauncher::run()
      │                             ├─ XeFMHAFwdSplitKVKernel  ← GPU: Split-KV 计算
      │                             └─ ReduceSplitK            ← GPU: 合并 splits
      │
      ▼
    返回 (out, softmax_lse)
```

### 5.2 GDN Attention 调用流程

```
Python: torch.ops._xpu_C.gdn_attention(...)
  │  [测试调用 / vLLM 集成]
  │
  ▼
C++ Op: gdn_attention()
  │  [csrc/xpu/gdn_attn/gdn_attn_interface.cpp]
  │
  │  输入验证 (连续性, shape)
  │  解析 activation mode (silu/swish)
  │
  ├── if VLLM_XPU_ENABLE_XE2 && num_prefills > 0
  │   │
  │   │  分配带 padding 的 Q, K, V, B, A (zeros)
  │   │  padding_size = batch_size * (chunk_size - 1)
  │   │
  │   ├── chunk_causal_conv1d_xe2()
  │   │   │  分块因果卷积
  │   │   │  处理 conv_state 更新
  │   │   └── 输出 Q, K, V, Z, B, A
  │   │
  │   └── chunk_gated_delta_rule_xe2()
  │       │  分块门控增量规则
  │       │  利用 DPAS 加速矩阵运算
  │       └── 输出 core_attn_out, 更新 ssm_state
  │
  └── else (通用路径)
      │
      │  分配 Q, K, V, B, A (empty)
      │
      ├── gdn::causal_conv1d()
      │   │  [csrc/xpu/gdn_attn/causal_conv1d.hpp]
      │   │  标准 SYCL 因果卷积
      │   └── 输出 Q, K, V, Z, B, A
      │
      └── gdn::gated_delta_rule()
          │  [csrc/xpu/gdn_attn/gated_delta_rule.hpp]
          │  标准 SYCL 门控增量规则
          └── 输出 core_attn_out, 更新 ssm_state
```

---

## 6. Kernel 选择逻辑与 Policy 分发

### 6.1 Head Size 策略分发

**Chunk Prefill（5 级分发）：**

```cpp
if (head_size <= 64)   → chunk_policy_head64   // 8 SGs
if (head_size <= 96)   → chunk_policy_head96   // 8 SGs
if (head_size <= 128)  → chunk_policy_head128  // 16 SGs
if (head_size <= 192)  → chunk_policy_head192  // 32 SGs
if (head_size <= 256)  → chunk_policy_head256  // 32 SGs
```

**Paged Decode（5 × 2 × 2 分发）：**

```cpp
// Step 1: Q Group Size 分发
if (num_q_group_size <= 8)   → dispatch_by_page_size<_8>(...)
if (num_q_group_size <= 16)  → dispatch_by_page_size<_16>(...)

// Step 2: Page Size 分发 (64 或 128)
switch (page_size) {
    case 64:  → dispatch_by_head_size<QGroup, _64>(...)
    case 128: → dispatch_by_head_size<QGroup, _128>(...)
}

// Step 3: Head Size 分发 (与 prefill 相同的 5 级)
```

### 6.2 数据类型分发

每个 policy 支持 6 种 Q/K 类型组合：

| Q 类型 | K/V 类型 | 说明 |
|--------|----------|------|
| half | half | 标准 FP16 |
| half | float8_e4m3 | FP16 query + FP8 KV (e4m3) |
| half | float8_e5m2 | FP16 query + FP8 KV (e5m2) |
| bfloat16 | bfloat16 | 标准 BF16 |
| bfloat16 | float8_e4m3 | BF16 query + FP8 KV (e4m3) |
| bfloat16 | float8_e5m2 | BF16 query + FP8 KV (e5m2) |

### 6.3 Boolean 特性组合分发

**Chunk Prefill** 使用递归模板展开 4 个 bool 参数：

```
policy_dispatch_func<Policy>(queue, cuQKType, args, is_paged, is_causal, is_local, is_sink)
                                                     │         │         │         │
递归展开：                                             ▼         ▼         ▼         ▼
policy_dispatch_impl<Policy, Paged, Causal, Local, Sink>(...)

总共 2^4 = 16 种组合 × 5 种 head size × 6 种 dtype = 480 个实例化
```

**Paged Decode** 展开 3 个 bool 参数：

```
decode_policy_dispatch_func<Policy>(queue, cuQKType, args, is_causal, is_local, is_sink)

总共 2^3 = 8 种组合 × 20 种 policy × 6 种 dtype = 960 个实例化
```

**编译优化：** 使用 extern template + 显式实例化（CMake 生成的 .cpp 文件），避免每个翻译单元重复编译。

### 6.4 Decode 策略的额外维度

Decode kernel 比 Prefill 多一个 **Page Size** 维度：

```
Prefill: head_size → Policy
Decode:  (head_size, q_group_size, page_size) → Policy
```

这是因为 Decode 模式必须处理 paged KV cache，page size 影响了 KV tile 的对齐和加载策略。

---

## 7. GPU 硬件适配与架构约束

### 7.1 架构检查

```cpp
// attn_interface.cpp
if (vllm::xpu::is_xe2_arch()) {
    // 使用 XE2 CUTLASS kernel
} else {
    TORCH_CHECK(false, "Only XE2 cutlass kernel is supported currently.");
}
```

**当前仅支持 XE2 架构。** 非 XE2 架构会直接报错。

### 7.2 XE2 硬件特性利用

| 特性 | 用途 |
|------|------|
| DPAS（点积累加系统阵列） | MMA 操作（Q×K, P×V 矩阵乘） |
| Block 2D Load | 高效矩阵加载 |
| Subgroup Size = 16 | CUTLASS kernel 的 SIMD 宽度 |
| GRF Size = 256 | 大量寄存器文件 |
| Shared Local Memory | Work group 内数据共享 |
| LSC (Load/Store Cache) | 内存访问优化 |

### 7.3 num_splits 的硬件感知计算

```cpp
int num_xe_cores =
    device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
    device.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
```

通过查询实际硬件的 XE Core 数量，动态调整 KV split 数以最大化 GPU 利用率。

---

## 8. 支持的特性矩阵

### FlashAttention v2

| 特性 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| Variable Length | ✅ | ✅ | 通过 cu_seqlens_q 支持不同长度序列 |
| Paged KV Cache | ✅ | ✅(必须) | 通过 block_table 映射 |
| Causal Mask | ✅ | ❌(强制关闭) | decode 时无需 causal |
| Local/Sliding Window | ✅ | ✅ | window_size_left / right |
| Softmax Sink | ✅ | ✅ | 额外的 sink token 注意力 |
| FP8 KV Cache (e4m3) | ✅ | ✅ | 需要 k_scale + v_scale |
| FP8 KV Cache (e5m2) | ✅ | ✅ | 需要 k_scale + v_scale |
| GQA (Grouped Query) | ✅ | ✅ | head_group = num_heads_q / num_heads_kv |
| MQA (Multi-Query) | ✅ | ✅ | GQA 的特殊情况 |
| Head Size 64-256 | ✅ | ✅ | 5 级策略 |
| BF16/FP16 | ✅ | ✅ | Q 必须是 bf16/fp16 |
| Split-KV | ❌ | ✅ | 自动计算 num_splits |
| Dropout | ❌ | ❌ | 推理时不使用 |
| Softcap | ❌(参数存在) | ❌ | 虽然接口有但内部未实现 |
| ALiBi Slopes | ❌(参数存在) | ❌ | 接口有但未实现 |
| Return Softmax LSE | 部分 | 部分 | 当前返回空 tensor |

### GDN Attention

| 特性 | 通用路径 | XE2 加速 | 说明 |
|------|----------|----------|------|
| Prefill | ✅ | ✅ | 多 token 处理 |
| Decode | ✅ | ❌(回退通用) | 单 token 处理 |
| Mixed Mode | ✅ | 部分 | Prefill 走 XE2, decode 走通用 |
| Conv State | ✅ | ✅ | 增量推理状态 |
| SSM State | ✅ | ✅ | 门控增量规则状态 |
| SiLU/Swish | ✅ | ✅ | 激活函数 |
| Conv Bias | ✅ | ✅ | 可选 |
| FP16 | ✅ | ✅ | |
| BF16 | ✅ | ✅ | |
| FP32 | ✅ | — | |
| Tensor Parallelism | ✅ | ✅ | tp_size 参数 |
| Chunk Size | — | 64 | XE2 固定 |

---

## 9. 关键数据结构与张量布局

### 9.1 FlashAttention 张量布局

```
Query:       [total_seq, num_heads_q, head_size]    # ragged format
Key (paged): [num_blocks, block_size, num_heads_kv, head_size]
Key (dense): [total_seq, num_heads_kv, head_size]
Value:       与 Key 相同布局
Output:      [total_seq, num_heads_q, head_size]

Block Table: [batch_size, max_num_blocks_per_seq]   # int32
cu_seqlens_q: [batch_size + 1]                      # int32, 累积长度
cu_seqlens_k: [batch_size + 1] 或 seqused_k: [batch_size]  # int32

# Decode 额外张量
tmp_out:     [num_tokens, num_heads_q * num_kv_splits, head_dim]
max_logits:  [num_tokens, num_heads_q, num_kv_splits]   # float32
exp_sums:    [num_tokens, num_heads_q, num_kv_splits]   # float32
```

### 9.2 GDN Attention 张量布局

```
projected_states_qkvz: [total_seqlen, num_k_heads/tp * (2*head_k_dim + 2*head_v_dim*num_v_heads/num_k_heads)]
projected_states_ba:   [total_seqlen, num_k_heads/tp * (2*num_v_heads/num_k_heads)]

conv_state:    [cache_batch_size, width-1, mixed_qkv_size]
ssm_state:     [cache_batch_size, num_v_heads/tp, head_v_dim, head_k_dim]
conv_weights:  [mixed_qkv_size, width]
conv_bias:     [mixed_qkv_size]  (optional)

core_attn_out: [total_seqlen, num_v_heads/tp, head_v_dim]
z:             [total_seqlen, num_v_heads/tp, head_v_dim]
```

### 9.3 CUTLASS 内部 Stride 约定

```cpp
StrideQ = Stride<int, _1, int, int>   // [seq, head_dim, heads, batch]
StrideK = Stride<int, _1, int, int>   // [seq, head_dim, heads, batch]
StrideV = Stride<_1, int, int, int>   // [head_dim, seq, heads, batch]  # 注意 V 转置
StrideO = Stride<int, _1, int, int>   // [seq, head_dim, heads, batch]
```

注意 V 的 stride 布局与 Q/K/O 不同（第一维是 `_1`），这意味着 V 的 head_dim 维度是连续的（转置布局），这是因为 PV GEMM 需要这种布局。

---

## 10. 编译与构建体系

### 10.1 CMake 构建

```python
# setup.py 中定义的额外库
additional_libraries = {
    "attn_kernels_xe_2": "/csrc/xpu/attn/xe_2",           # FlashAttention XE2 kernel
    "gdn_attn_kernels_xe_2": "/csrc/xpu/gdn_attn/xe_2",   # GDN Attention XE2 kernel
    "grouped_gemm_xe_default": "...",
    "grouped_gemm_xe_2": "...",
}
```

### 10.2 模板实例化策略

为避免编译时间爆炸，项目使用了 **extern template + CMake 生成的显式实例化** 策略：

1. **chunk_prefill_extern.hpp**：声明 5 × 16 = 80 个 extern template
2. **paged_decode_extern.hpp**：声明 20 × 8 = 160 个 extern template
3. CMake 为每个实例化生成独立的 .cpp 文件

X-Macro 模式用于自动生成所有组合：

```cpp
#define CHUNK_POLICY_LIST(X) \
  X(chunk_policy_head64)     \
  X(chunk_policy_head96)     \
  X(chunk_policy_head128)    \
  X(chunk_policy_head192)    \
  X(chunk_policy_head256)

// 对每个 policy 生成所有 2^4 = 16 个 bool 组合的 extern template 声明
CHUNK_POLICY_LIST(DECLARE_ALL_BOOL_COMBINATIONS)
```

### 10.3 编译条件宏

| 宏 | 含义 |
|----|------|
| `VLLM_XPU_ENABLE_XE2` | 启用 XE2 架构特定 kernel |
| `FLASH_NAMESPACE` | FlashAttention 命名空间 |
| `TORCH_EXTENSION_NAME` | PyTorch 扩展模块名 |

---

## 11. 测试体系分析

### 11.1 FlashAttention 测试

**文件：** `tests/flash_attn/test_flash_attn_varlen_func.py`

**两个测试函数：**

1. **`test_varlen_with_paged_kv`**：Prefill + 可选 Paged 模式
   - 固定序列 `[(1, 1328), (5, 18), (129, 463)]` - 混合长短序列
   - 参数空间：4(heads) × 4(head_size) × 2(block_size) × 2(dtype) × 4(window) × 2(sink) × 2(casual) × 2(paged) × 3(fp8) × 2(num_blocks) = 12,288 组合

2. **`test_decode_with_paged_kv`**：Decode 模式（必须 paged）
   - 序列 `[(1, x)]` 格式，多种长度
   - 参数空间：4(seqs) × 4(heads) × 4(head_size) × 2(block_size) × 2(dtype) × 2(num_blocks) × 2(sink) × 3(fp8) × 4(window) = 24,576 组合

**参考实现：** `ref_paged_attn` 使用纯 PyTorch 实现标准 attention 作为金标准。

**精度要求：**
- 标准：atol=1e-2, rtol=1e-2
- 量化输入：atol=1.5e-1, rtol=1.5e-1
- 滑动窗口/FP8：atol=1.5e-2, rtol=1.5e-2

### 11.2 GDN Attention 测试

**文件：** `tests/gdn_attn/test_gdn_attn.py`

**参数空间：**
- num_tokens: [1, 32, 1024, 8192]
- batch_size: [32]
- num_k/v_heads: [16, 32]
- head_dim: [128]
- mode: [prefill, decode, mix_mode]
- dtype: [float16]

**参考实现：** `ref_gdn_attention` 完全用 Python 实现 GDN 状态机逻辑。

**精度要求：** atol=5e-2, rtol=5e-2

### 11.3 Mini Pytest 模式

两个测试文件都支持 `MINI_PYTEST_PARAMS` 配置，用于 CI 中运行精简测试集。

---

## 12. 重要发现与设计亮点

### 12.1 Decode 阶段 Causal Mask 的正确处理

在 `flash_api.cpp` 中有一个精妙的设计决策：**decode 阶段强制关闭 causal mask**。

```cpp
// For paged decode (single query per sequence), causal masking is a no-op:
// seqused_k already constrains KV to only the valid past tokens,
// so there are no "future" tokens to mask.
cutlass_paged_decode_interface(..., false /* is_causal */);
```

这是因为 decode 时 `seqused_k` 已经限制了 KV 的有效范围，causal mask 会引入错误的 seq_len 计算。

### 12.2 Split-KV 的硬件感知自适应

`get_num_splits` 函数根据实际 GPU 硬件的 XE Core 数量动态调整 split 数，而非使用固定值。这确保了在不同 Intel GPU 型号上都能获得最优性能。

### 12.3 Extern Template 优化编译时间

面对 480+960 = 1440 个 kernel 实例化，项目使用 extern template 机制 + CMake 自动生成，将编译分散到独立的翻译单元中，显著降低了增量编译时间和内存占用。

### 12.4 V 张量的转置布局

CUTLASS kernel 中 V 使用转置布局 `Stride<_1, int, int, int>`（head_dim 连续），这与 Q/K 的 `Stride<int, _1, int, int>`（head_dim 连续在第二维）不同。这个设计是为了优化 P×V 矩阵乘中的内存访问模式。

### 12.5 GDN 的 XE2 选择性加速

GDN Attention 只在 **prefill 模式** 使用 XE2 加速，decode 模式回退到通用 SYCL kernel。这是因为：
- Prefill 处理多个 token，可以利用 chunk 并行性
- Decode 只处理单个 token，chunk 化没有收益
- 通用 kernel 对单 token 场景已足够高效

### 12.6 Softmax Sink 机制

FlashAttention kernel 支持 "softmax sink" 机制，为每个 attention head 添加一个额外的可学习标量 sink。这用于处理某些模型中 attention score 需要一个 "default" 目标的场景：

```cpp
if (Sink) {
    ElementSink s_head = p.ptr_S[head_q];
    epilogue(O, tArA, tA_max, tA_sum, blk_qv, s_head, thr_id);
}
```

### 12.7 PyTorch Shim 类型适配

`pytorch_shim.h` 的设计非常优雅，通过模板元编程自动解决 FlashAttention 原始 C++ API 与 PyTorch library 绑定之间的类型不兼容问题，使得对上游代码的侵入性最小。

### 12.8 Window Size 与 Causal 的交互

当同时启用 local window 和 causal mask 时，causal 被转换为 local：

```cpp
if (is_local) {
    if (is_causal) {
        window_size_right = 0;   // causal = 只看左边
        is_causal = false;       // 由 local 机制实现
    }
}
```

---

## 13. 附录：文件索引

### FlashAttention v2 文件

| 文件路径 | 功能 |
|----------|------|
| `vllm_xpu_kernels/flash_attn_interface.py` | Python 层接口 |
| `csrc/flash_attn/flash_api.cpp` | C++ API + Prefill/Decode 路由 |
| `csrc/flash_attn/pytorch_shim.h` | PyTorch 类型适配垫片 |
| `csrc/xpu/attn/attn_interface.cpp` | 架构分发层 |
| `csrc/xpu/attn/attn_interface.h` | 接口声明 |
| `csrc/xpu/attn/xe_2/fmha_utils.hpp` | Tile Policy 定义 + 数据类型映射 |
| `csrc/xpu/attn/xe_2/fmha_xe2.cpp` | Chunk Prefill 实现 |
| `csrc/xpu/attn/xe_2/fmha_xe2.h` | Chunk Prefill 声明 |
| `csrc/xpu/attn/xe_2/chunk_prefill.hpp` | Prefill Kernel 配置 + Policy 分发 |
| `csrc/xpu/attn/xe_2/chunk_prefill_utils.hpp` | Policy 分发辅助函数 |
| `csrc/xpu/attn/xe_2/chunk_prefill_extern.hpp` | Extern template 声明 |
| `csrc/xpu/attn/xe_2/paged_decode.hpp` | Decode Kernel 配置 + Policy 分发 |
| `csrc/xpu/attn/xe_2/paged_decode_utils.hpp` | Decode 分发辅助函数 |
| `csrc/xpu/attn/xe_2/paged_decode_extern.hpp` | Decode Extern template 声明 |
| `csrc/xpu/attn/xe_2/paged_decode_xe2.cpp` | Paged Decode 实现 |
| `csrc/xpu/attn/xe_2/paged_decode_xe2.h` | Paged Decode 声明 |
| `csrc/xpu/attn/xe_2/kernel/chunk_prefill_kernel.hpp` | Prefill CUTLASS Kernel |
| `csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp` | Decode CUTLASS Kernel + ReduceSplitK |
| `csrc/xpu/attn/xe_2/collective/chunk_prefill_mainloop.hpp` | Prefill Mainloop (QK+Softmax+PV) |
| `csrc/xpu/attn/xe_2/collective/chunk_prefill_epilogue.hpp` | Epilogue (Output + Sink) |
| `csrc/xpu/attn/xe_2/collective/chunk_prefill_scheduler.hpp` | Tile Schedulers |

### GDN Attention 文件

| 文件路径 | 功能 |
|----------|------|
| `csrc/xpu/gdn_attn/gdn_attn_interface.cpp` | C++ 入口 + 路径选择 |
| `csrc/xpu/gdn_attn/gdn_attn_utils.h` | GDN 工具定义 (ActMode, chunk_size) |
| `csrc/xpu/gdn_attn/causal_conv1d.hpp` | 因果 Conv1d SYCL Kernel |
| `csrc/xpu/gdn_attn/gated_delta_rule.hpp` | 门控增量规则 SYCL Kernel |
| `csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp` | XE2 优化的分块因果卷积 |
| `csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_xe2.cpp` | XE2 优化的分块 GDN |
| `csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_xe2.h` | XE2 GDN 声明 |
| `csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp` | XE2 GDN GEMM Kernel |
| `csrc/xpu/gdn_attn/xe_2/gemm.hpp` | 底层 GEMM 操作 |

### 测试文件

| 文件路径 | 功能 |
|----------|------|
| `tests/flash_attn/test_flash_attn_varlen_func.py` | FlashAttention 功能测试 |
| `tests/gdn_attn/test_gdn_attn.py` | GDN Attention 功能测试 |

### 绑定与注册

| 文件路径 | 功能 |
|----------|------|
| `csrc/xpu/torch_bindings.cpp` | _xpu_C 算子注册 (含 gdn_attention) |
| `csrc/flash_attn/flash_api.cpp` | _vllm_fa2_C 算子注册 |

---

*报告生成时间：2026-03-12*
*项目版本：基于 hongbolv/vllm-xpu-kernels 仓库最新代码*
