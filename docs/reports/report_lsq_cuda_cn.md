# LSQ 复现与 CUDA 系统优化项目汇报

## 1. 项目背景

我最开始做的是 LSQ 论文复现，目标是验证低比特量化训练在 ImageNet 上的效果。完成复现后，我带着这个项目去和老师交流。老师给我的建议是，不要只停留在算法复现层面，而要进一步关注系统优化，因为低比特算法本身并不等价于低比特执行效率。

当时我也观察到一个直接现象：在我的实验里，LSQ 4-bit 的训练时间反而大于 LSQ 8-bit 和 FP baseline。老师指出，这背后的核心原因是当前 GPU 并不会自动为 4-bit 提供高效执行路径，因此低比特模型在通用框架中不一定更快。基于这个反馈，我把项目进一步推进到了“量化训练结果如何映射到硬件友好的 CUDA 推理实现”这个方向。

## 2. 项目目标

这个项目目前的目标可以概括为两部分：

1. 先完成 LSQ 论文复现，拿到可信的低比特模型训练结果。
2. 再将训练好的 LSQ checkpoint 接入自定义 CUDA 低比特推理路径，验证：
   - 量化模型是否能与自定义低比特表示兼容
   - 整网精度是否能够保持
   - 整网推理瓶颈到底在哪里
   - 哪些系统层面的优化是有效的

## 3. 项目推进历程

### 阶段 1：LSQ 论文复现

我在 `/home/yph3738/projects/ece_9483/LSQ` 中完成了 LSQ 的实现和训练流程，主要包括：

- pre-activation ResNet-18
- LSQ quantizer
- ImageNet 训练和验证
- FP baseline 与 LSQ fine-tuning 流程

复现得到的核心结果是：

- FP baseline Top-1 约为 `69.67`
- LSQ W4A4 模型 Top-1 约为 `68.6+`
- 量化模型与 FP 模型相比只下降约 1 个点，说明复现结果可信

在这个阶段，我还进一步检查并修正了 LSQ 实现中的 step-size gradient scaling 细节，使其更接近论文附录中的 `NW/NF` 设定。

### 阶段 2：从算法复现转向系统问题

在与老师交流后，我开始把问题从“LSQ 训练精度”转向“LSQ 模型如何高效执行”。

一个关键认识是：

- 低比特训练结果并不自动对应低比特执行效率
- 当前通用 GPU 路径不会自动把 LSQ 的 4-bit 权重映射成真正高效的 4-bit 运算
- 因此需要显式设计低比特数据表示和 CUDA kernel

### 阶段 3：搭建 CUDA 优化项目

我在 `/home/yph3738/projects/cuda_optimization` 中搭建了一个低比特 CUDA 推理原型，主要实现了：

- INT4 权重打包表示
- INT4-storage / INT8-compute CUDA GEMM kernel
- 外部 LSQ checkpoint adapter
- `fc` 层 benchmark
- native LSQ 与 converted model 的整网评测脚本

这个阶段的意义在于：我不再只是停留在“训练出一个量化模型”，而是把量化模型真正接进了一个硬件友好的低比特执行链路。

### 阶段 4：整网精度验证

为了验证自定义低比特路径是否只是一个 layer demo，我进一步做了整网 ImageNet 验证。

在 `paper-style LSQ W4A4, first/last 8-bit` 设置下：

- 中间 19 个量化卷积层被成功替换
- `conv1` 和 `fc` 因为仍保持 8-bit，没有替换
- converted model 的精度与 native model 基本一致

这一步说明：

- 自定义低比特路径在整网层面是可用的
- 当前系统优化虽然还不一定更快，但至少已经证明了端到端功能和精度是成立的

### 阶段 5：profiling 分析系统瓶颈

接下来我没有继续盲目优化 kernel，而是先做了 profiling。

profiling 的核心发现是：

- 整网慢的原因并不主要是量化算法本身
- 也不只是 GEMM kernel 算得不够快
- 真正的瓶颈主要来自：
  - `F.unfold` 的 lowering 开销
  - 大量中间张量分配
  - dtype 转换
  - 碎片化的 kernel launch

这个结论很关键，因为它说明整网性能瓶颈是“卷积 lowering / integration strategy”的问题，而不是“LSQ 算法本身”的问题。

### 阶段 6：针对热点路径做系统优化

基于 profiling 结果，我没有再继续做无针对性的改动，而是逐步对热点路径进行优化：

1. fixed-batch profiling  
   先固定 GPU batch，只测模型前向，去掉 DataLoader 干扰。

2. 减少中间分配和无效转换  
   做了 buffer reuse、减少中间 copy 和 dtype 往返。

3. `1x1 conv` 特化  
   对 pointwise conv 做更直接的 lowering，作为小规模试探。

4. 针对 `3x3, stride=1, padding=1` 写了 fused CUDA lowering path  
   这是最关键的一步。原本 `3x3` conv 走的是 Python-side `F.unfold + GEMM`，我把 quantize + lowering 合并进了 CUDA。

5. 去掉 `.item()` 导致的 host sync  
   把 scale 缓存在 host 侧，避免每次 forward 做 GPU 到 CPU 标量同步。

6. 让 `3x3 lowering kernel` 直接输出 padded K 维  
   进一步减少 Python 侧 pad/copy 操作。

这条优化路线已经不再是“零散改几行代码”，而是一个很明确的系统优化思路：

- 先定位瓶颈
- 再把高频热点从 Python/PyTorch op 下沉到 CUDA
- 逐步减少 lowering、allocation、sync 和 kernel launch

## 4. 核心实验结果汇总

| 实验项 | 结果 | 说明 |
|---|---:|---|
| LSQ baseline eval | Top-1 `68.58`, Top-5 `88.25` | 在 LSQ 原始仓库 `eval.py` 中复测 |
| native full-model eval | Top-1 `68.69`, Top-5 `88.30`, Time `57.4s` | `cuda_optimization` 中 native 路径整网评测 |
| converted full-model eval | Top-1 `68.75`, Top-5 `88.38`, Time `148.5s` | 初始整网转换路径，精度保持但速度更慢 |
| converted layer count | `19` | 成功替换的中间量化 conv 层数量 |
| skipped layer count | `2` | `conv1` 和 `fc` 保持 8-bit，未替换 |
| fixed-batch profile baseline converted | Wall time `15.44s` | 初始 converted profiling |
| after 3x3 fused lowering kernel | Wall time `9.62s` | 将 `3x3 s1 p1` quantize+lowering 下沉到 CUDA 后 |
| after removing `.item()` + padded lowering | Wall time `8.46s` | 当前最新 profiling 结果 |

## 5. 结果解读

目前的结果可以得出几个明确结论：

1. LSQ 复现本身是成功的。4-bit 模型在 ImageNet 上与 FP baseline 只差约 1 个点，基线可信。
2. 自定义低比特 CUDA 路径在整网层面已经可用。converted model 的 Top-1/Top-5 与 native model 基本一致，说明端到端精度保持成立。
3. 整网初始不加速，不代表方向错误。最初 converted 整网推理比 native 更慢，并不是因为量化方法没意义，而是因为卷积 lowering 和大量中间操作把低比特计算的潜在收益吃掉了。
4. profiling 驱动的系统优化是有效的。通过把 `3x3` conv 的 quantize+lowering 从 Python-side `F.unfold` 下沉到 CUDA，并进一步去掉 host sync 和额外 pad/copy，fixed-batch profiling wall time 从 `15.44s` 降到了 `8.46s`。

## 6. 当前局限

目前仍然存在几个明显限制：

- `conv1` 和 `fc` 仍然保持 8-bit，未纳入当前 INT4 路径
- converted path 仍有较多 kernel launch
- `clamp/div/round` 等前处理还未完全融合
- 当前还没有形成真正意义上的 fused conv kernel，仍然属于“specialized lowering + GEMM”阶段
- 整网速度目前仍未超过 cuDNN native 路径

所以当前最准确的定位是：这是一个已经验证可行性的低比特系统优化原型，而不是最终的高性能推理框架。

## 7. 下一步计划

下一步最值得推进的方向是：

1. 继续减少碎片化 kernel launch，尝试进一步融合 quantization 前处理和后处理。
2. 进一步面向 `3x3 conv` 做 specialized 优化，目标是继续减少 lowering 和中间张量开销。
3. 从显式 lowering 向 implicit GEMM 或 fused conv 路径过渡，这是最自然的下一阶段系统优化方向。

## 8. 总结

这个项目目前已经形成了一条完整的技术路线：

- 我先完成了 LSQ 论文复现，获得了可信的低比特训练基线；
- 然后把训练好的 LSQ checkpoint 接入自定义的 INT4-storage / INT8-compute CUDA 推理路径；
- 在整网层面验证了转换后的模型能够保持精度；
- 通过 profiling 发现整网性能瓶颈主要来自卷积 lowering 和大量碎 kernel，而不是量化算法本身；
- 针对 `3x3` conv 的 quantize+lowering 下沉到 CUDA后，fixed-batch profile 时间从 `15.44s` 降至 `8.46s`，说明系统优化方向是有效的。

如果后续继续推进，我希望把这个项目从“低比特推理原型”进一步推进到“更接近真实部署效率的低比特系统实现”。
