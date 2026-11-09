+++
title = "ViT 与 Swin Transformer 学习实践"
slug = "vit-swin-transformer"
date = "2026-11-09T16:20:00+08:00"
draft = false
author = "Llj159-lab"
description = "学习 Vision Transformer 与 Swin Transformer 的网络结构与高效注意力机制，并完成相关思考题。"
tags = ["深度学习", "Transformer", "ViT", "Swin Transformer", "注意力机制"]
categories = ["课程实验"]
showToc = true
+++

<style>
p {
  text-indent: 2em;
}

table {
  margin-left: auto;
  margin-right: auto;
}
</style>

<div align="center">

# 实验报告：ViT & Swin Transformer
</div>


## 一、实验信息

| 项目 | 内容 |
| --- | --- |
| 实验名称 | 实验5：ViT & Swin Transformer |
| 实验平台 | Google Colab、Python、PyTorch |

## 二、实验目的

1. 理解 Transformer 的基本原理，及其在计算机视觉任务中的应用。
2. 掌握 Vision Transformer（ViT）的网络结构，包括 Patch Embedding、Transformer Encoder、MLP Head。
3. 了解 Swin Transformer 的层次化结构与 Window Attention / Shifted Window Attention 机制。
4. 了解视觉 Transformer 的发展脉络（综述视频），把握高效自注意力的主流设计思路。
5. 通过思考题加深对高效注意力机制、CNN 与 Transformer 结合方式、以及 Transformer 结构改进等前沿问题的理解。

## 三、实验内容与学习要求

- 学习 Transformer 的基本原理。
- 学习典型基于 Transformer 的网络结构（ViT、Swin Transformer）。
- 观看视频学习资料，并完成思考题。

## 四、视频学习总结

### 4.1 Vision Transformer（ViT）

ViT 将图像划分为固定大小的 patch，把每个 patch 展平并线性映射为 token，加上位置编码后送入标准的 Transformer Encoder，最后通过 MLP Head 完成分类。其主要组件如下：

1. **Patch Embedding**：将大小为 H×W×C 的图像切分成 N 个 P×P 的 patch（N = HW/P²），每个 patch 展平后经线性层映射为 D 维向量。该过程等价于用卷积核大小和步长均为 P 的卷积实现。此外，附加一个可学习的 `[class]` token 用于分类。
2. **Position Embedding**：引入可学习的位置编码，与 token 相加，以保留空间位置信息（self-attention 本身对顺序不敏感）。
3. **Transformer Encoder**：由 L 层堆叠而成，每层包含 Multi-head Self-Attention（MSA）、LayerNorm 和 MLP，并带有残差连接。
4. **MLP Head**：取出 `[class]` token 对应的输出，经 LayerNorm 后送入分类头，得到最终分类结果。

ViT 的注意力复杂度为 O(N²)，N 为 patch（token）数量，当图像分辨率增大时计算开销会显著增加。

### 4.2 Swin Transformer

Swin Transformer 引入了层次化（hierarchical）结构和基于窗口（window）的注意力机制，使其具备类似 CNN 的多尺度特征表达能力。关键设计如下：

1. **Patch Partition / Patch Merging**：输入被划分为 4×4 的 patch；随着网络加深，Patch Merging 将相邻 2×2 的 patch 合并，使特征图分辨率逐层降低、通道数逐层加倍，形成金字塔式特征。
2. **W-MSA（Window-based Multi-head Self-Attention）**：将特征图划分为互不重叠的窗口，只在每个窗口内部计算自注意力，把计算量从 O(N²) 降为与窗口数量及窗口边长平方相关的线性量级。
3. **SW-MSA（Shifted Window MSA）**：相邻层之间将窗口整体平移（下移、右移半个窗口大小），实现跨窗口的信息交互，弥补 W-MSA 缺乏全局连接的不足。
4. **Relative Position Bias**：在注意力计算中引入相对位置偏置，增强位置建模能力。

### 4.3 视觉 Transformer 综述

梳理了 ViT 的发展脉络，主要包括：原生 ViT、层次化 ViT（Swin、PVT 等）、CNN 与 Transformer 的混合结构，以及自注意力的高效化设计（窗口注意力、空间降采样注意力、线性注意力等）。

## 五、思考题解答

### 5.1 在 ViT 中要降低 Attention 的计算量，有哪些方法？

标准自注意力的复杂度为 O(N²d)，其中 N 为 token（patch）数，d 为特征维度。随着图像分辨率增大，N 快速增长，计算量成为主要瓶颈。降低注意力计算量的方法主要有以下几类：

| 方法 | 代表工作 | 核心思路 |
| --- | --- | --- |
| 窗口化局部注意力 | Swin | 在局部窗口内计算 attention，复杂度从 O(N²) 降为 O(N·W²)；再通过 Shifted Window 实现跨窗口交互 |
| 空间降采样注意力 | PVT（Spatial Reduction Attention） | 对 K、V 先做卷积（stride>1）降采样，把空间尺寸缩小 R 倍，复杂度降为原来的 1/R² |
| 低秩近似 | Linformer | 利用注意力矩阵的低秩性，将 N×N 的注意力矩阵近似为 N×k（k 为常数），复杂度降为 O(N) |
| 核方法 / 线性注意力 | Performer、Linear Attention | 用核特征映射把 softmax(QKᵀ) 分解，改变乘法结合顺序，将复杂度线性化 |
| 稀疏 / 局部注意力 | 各类局部注意力方法 | 只对局部邻域或选定 token 计算注意力，减少参与计算的 token 对 |
| 分层降分辨率结构 | Swin、PVT | 逐层合并 patch（Patch Merging），在更粗的分辨率上计算注意力，减少 token 数并扩大感受野 |

其中，Swin 的 Window Attention 将注意力限制在互不重叠的局部窗口内，并用移位窗口解决窗口之间缺乏信息交互的问题；PVT 的 Spatial Reduction Attention（SRA） 则在生成 K、V 之前先进行卷积降采样，从而以较小的精度代价显著降低计算量。

### 5.2 Swin 体现了一种什么思路？对后来工作有哪些启发？

Swin Transformer 体现的核心思路是 “先局部、再整体”的层次化设计：

- **先局部**：通过 W-MSA 在每个局部窗口内部建模局部关系，控制计算量。
- **再整体**：通过 SW-MSA 实现相邻窗口之间的信息交互，再通过 Patch Merging 逐层扩大感受野，使深层网络逐步获得全局语义信息。
- **层次化**：分辨率逐层降低、通道数逐层增加，形成与 CNN 类似的“金字塔”特征结构。

这种设计对后来工作的启发主要体现在：

1. **证明 Transformer 可作为通用视觉 backbone**：Swin 的多尺度特征表达使其不仅能做分类，还能用于目标检测、语义分割等密集预测任务，突破了原生 ViT 单一尺度、全局注意力的局限。
2. **“局部注意力 + 跨窗口/跨层信息交换”成为高效视觉 Transformer 的通用范式**：后续大量工作（如 SwinV2、Twins、Focal Transformer 等）都延续了这一设计思想。
3. **重新引入 CNN 的归纳偏置**：Swin 将局部性、层次化、平移不变性等 CNN 归纳偏置重新引入 Transformer，说明纯全局注意力并非唯一选择，恰当的归纳偏置仍然是视觉建模的重要先验。

### 5.3 有些网络将 CNN 和 Transformer 结合，为什么一般把 CNN block 放在前面，Transformer block 放在后面？

将 CNN block 放在前面、Transformer block 放在后面，主要基于以下原因：

1. **分工互补**：CNN 擅长提取局部、底层的低级特征（边缘、纹理、颜色等），具有局部性、平移不变性等强归纳偏置，参数少、计算高效；Transformer 擅长建模长程依赖和全局上下文，更适合处理高层语义。底层用 CNN、高层用 Transformer 正好各取所长。
2. **降低 token 数量与计算量**：原始分辨率下像素/token 数量巨大，自注意力的 O(N²) 复杂度难以承受。先用 CNN 的卷积和池化做下采样，将特征图缩小后再切分为 token，可显著降低 Transformer 的计算量。
3. **提供更合适的输入表示**：CNN 前端可以把原始像素转换成结构化、低维、带有局部上下文信息的特征图，这些特征比原始像素更适合作为 Transformer 的 token 输入。
4. **符合视觉信息的处理层次**：视觉理解通常遵循“先提取局部结构、再建立全局关系”的顺序。底层需要强局部先验（CNN 天然具备），高层需要全局建模（Transformer 更擅长），因此 CNN 在前、Transformer 在后的结构符合认知规律。

### 5.4 Restormer 对 attention + FFN 分别做了哪些改进？

Transformer 的基本结构为 attention + FFN。Restormer（Restoration Transformer，CVPR 2022）面向图像复原任务（去噪、去雨、去模糊等），针对这两个基本模块分别做了改进：

**（1）MDTA（Multi-DConv Head Transposed Attention，多深度卷积头转置注意力）——对 attention 的改进**

- 传统 self-attention 在空间维度上计算 token 之间的相关性，复杂度为 O(N²)。Restormer 将注意力转置到通道维度上，计算通道之间的相关性（cross-covariance），复杂度从 O(N²) 降为 O(d²)（d 为通道数，通常远小于空间 token 数 N），因此更适合高分辨率图像。
- 由于通道维注意力缺少空间上下文，Restormer 在生成 Q、K、V 时引入 3×3 depth-wise convolution（深度可分离卷积）来编码空间邻域信息，弥补空间信息的不足。
- 具体做法：Q、K、V 由 1×1 conv + 3×3 DConv 生成，reshape 后在通道维做注意力（点积 + softmax），多头并行，最后用 1×1 conv 聚合输出。

**（2）GDFN（Gated-Dconv Feed-Forward Network，门控深度卷积前馈网络）——对 FFN 的改进**

- 传统 FFN 本质是两个 1×1 conv（MLP）加 GELU，主要做通道混合，缺乏空间建模能力。
- GDFN 引入门控机制：先用 1×1 conv 扩张通道数（如 γ 倍），再用 3×3 DConv 编码空间邻域信息；然后将结果分成两路，一路经过 GELU 作为门控信号，与另一路逐元素相乘，最后用 1×1 conv 压缩回原通道数。
- 门控结构控制了信息流，使网络可以有选择地保留有用的空间信息，增强了局部上下文建模能力。

**总结**：Restormer 用 MDTA 把注意力从“空间维”改为“通道维”并辅以深度卷积，解决了高分辨率下注意力计算量过大的问题；用 GDFN 给 FFN 加入门控与深度卷积，增强了前馈网络的空间建模能力。两者共同使 Transformer 高效适用于图像复原任务，并取得了当时的 SOTA 效果。

## 六、视频链接

1. ViT 网络详解（Embedding、Transformer Encoder、MLP Head）：<https://www.bilibili.com/video/BV1Jh411Y7WQ>
2. 使用 PyTorch 实现 ViT：<https://www.bilibili.com/video/BV1AL411W7dT>
3. Swin Transformer 网络详解：<https://www.bilibili.com/video/BV1pL4y1v7jC>
4. 使用 PyTorch 实现 Swin Transformer（了解即可）：<https://www.bilibili.com/video/BV1yg411K7Yc>
5. 视觉 Transformer 综述（华为韩凯）：<https://www.bilibili.com/video/BV1cr4y1m7tS>

## 七、实验总结与心得

本次实验系统地学习了 Transformer 在计算机视觉领域的应用。掌握了 ViT 的基本结构，理解了“图像切 patch → 线性嵌入 → 位置编码 → Transformer Encoder → MLP Head”的完整流程；随后学习了 Swin Transformer 通过窗口注意力、移位窗口和 Patch Merging 实现“先局部、后整体”的层次化建模，从而兼顾计算效率与全局语义。

我进一步认识到：注意力计算量的优化是视觉 Transformer 落地的关键问题，Swin 的窗口注意力、PVT 的空间降采样注意力以及线性注意力等方法从不同角度解决了这一难题；CNN 与 Transformer 的结合（CNN 在前、Transformer 在后）本质上是对“局部先验 + 全局建模”两种能力的分工与互补；Restormer 则通过对 attention 与 FFN 两个模块的针对性改造（MDTA 与 GDFN），展示了 Transformer 结构在图像复原任务中的强大潜力。

本次实验加深了对 Transformer 基本原理及其在视觉任务中应用的理解，也体会到网络结构设计背后“效率与精度平衡”“归纳偏置与全局建模结合”等思想的重要性。
