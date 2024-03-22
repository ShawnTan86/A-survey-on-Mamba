# A-survey-on-Mamba
📄✅❗️0️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣
## ☀️Mamba
- Linear-Time Sequence Modeling with Selective State Spaces <br>
Paper Link: [📄📄📄](https://arxiv.org/ftp/arxiv/papers/2312/2312.00752.pdf), Code：[✅✅✅](https://github.com/state-spaces/mamba)


## Improvements and Optimizations Based on Mamba
- 0️⃣1️⃣ **MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts** <br>
Paper Link: [📄📄📄](https://arxiv.org/pdf/2401.04081.pdf), Code：❗️❗️❗️ <br>
Summary: 这篇论文介绍了MambaByte，这是一种无需标记的、基于状态空间模型（SSM）的字节级语言模型，它通过自回归方式训练于字节序列上。MambaByte直接使用字节作为序列的基本单元。MambaByte在多个数据集上展示了与其他字节级模型相比的优越性能，并与最先进的基于子词的Transformer模型竞争，同时在推理速度上由于其线性扩展特性而受益。研究表明，MambaByte是一种有效的无需标记的语言建模方法，为未来的大型模型提供了一种可行的无需标记的语言建模可能性。

- 0️⃣2️⃣ **MambaByte: Token-free Selective State Space Model** <br>
Paper Link: [📄📄📄](https://arxiv.org/pdf/2401.13660.pdf), Code：[✅✅✅](https://github.com/lucidrains/MEGABYTE-pytorch) <br>
Summary: 这篇论文介绍了MambaByte，这是一种无需标记的、基于状态空间模型（SSM）的字节级语言模型，它通过自回归方式训练于字节序列上。MambaByte直接使用字节作为序列的基本单元。MambaByte在多个数据集上展示了与其他字节级模型相比的优越性能，并与最先进的基于子词的Transformer模型竞争，同时在推理速度上由于其线性扩展特性而受益。研究表明，MambaByte是一种有效的无需标记的语言建模方法，为未来的大型模型提供了一种可行的无需标记的语言建模可能性。

- 0️⃣3️⃣ **LOCOST: State-Space Models for Long Document Abstractive Summarization** <br>
Paper Link: [📄📄📄](https://arxiv.org/pdf/2401.17919.pdf), Code：[✅✅✅](https://github.com/flbbb/locost-summarization) <br>
Summary: 这篇论文提出了LOCOST，一种基于状态空间模型（SSMs）的编码器-解码器架构，LOCOST用一个Mamba模型作为编码器，将长文转换为一维序列，再使用一个Transformer模型作为解码器，用于处理长文本摘要任务。LOCOST的计算复杂度为O(L log L)，相较于基于稀疏注意力模式的现有模型，能够处理更长的序列，同时在训练和推理过程中节省大量内存。实验表明，LOCOST在长文档摘要任务上的性能与最先进的稀疏变换器相当，同时在处理超过600K令牌的输入时，取得了新的最先进结果，为长文本处理开辟了新的可能性。

- 0️⃣4️⃣ **Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data** <br>
Paper Link: [📄📄📄](https://arxiv.org/pdf/2402.05892.pdf), Code：❗️❗️❗️ <br>
Summary: 这篇论文介绍了Mamba-ND，这是一种针对多维数据（如图像、视频等）的新型神经网络架构。Mamba-ND基于状态空间模型（SSM），通过在不同维度上交替处理输入数据，实现了与Transformer模型相当的性能，同时显著降低了参数数量和计算复杂度。在多个基准测试中，包括ImageNet-1K分类、HMDB-51和UCF-101动作识别、ERA5天气预测和BTCV 3D分割，Mamba-ND展现了与最先进模型相竞争的性能，并保持了线性复杂度。

- 0️⃣5️⃣ **Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling** <br>
Paper Link: [📄📄📄](https://arxiv.org/pdf/2402.10211.pdf), Code：[✅✅✅](https://github.com/raunaqbhirangi/hiss/tree/main) <br>
Summary: 这篇论文介绍了一种新的连续序列到序列建模技术，名为层次化状态空间模型（HiSS），它通过在不同时间分辨率上堆叠结构化的状态空间模型来创建时间层次结构。HiSS在六个真实世界的传感器数据集上的表现超越了现有的最先进序列模型，如因果Transformer、LSTM、S4和Mamba，至少在均方误差（MSE）上提高了23%。此外，实验表明HiSS在小数据集上具有高效的扩展性，并且与现有的数据过滤技术兼容。论文还发布了CSP-Bench，这是一个公共的连续序列预测基准，包含六个真实世界标记数据集，旨在支持多样化的感官数据分析。

- 0️⃣6️⃣ **Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks** <br>
Paper Link: [📄📄📄](https://arxiv.org/pdf/2402.04248.pdf), Code：❗️❗️❗️ <br>
Summary: 这篇论文介绍了一种新的连续序列到序列建模技术，名为层次化状态空间模型（HiSS），它通过在不同时间分辨率上堆叠结构化的状态空间模型来创建时间层次结构。HiSS在六个真实世界的传感器数据集上的表现超越了现有的最先进序列模型，如因果Transformer、LSTM、S4和Mamba，至少在均方误差（MSE）上提高了23%。此外，实验表明HiSS在小数据集上具有高效的扩展性，并且与现有的数据过滤技术兼容。论文还发布了CSP-Bench，这是一个公共的连续序列预测基准，包含六个真实世界标记数据集，旨在支持多样化的感官数据分析。

## ⭐️Vision Mamba
- 0️⃣1️⃣ **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model ** <br>
Paper Link: [📄📄📄](https://arxiv.org/pdf/2401.09417.pdf), Code：[✅✅✅](https://github.com/hustvl/Vim) <br>
Summary: 这篇论文提出了Vision Mamba (Vim)，一种基于双向状态空间模型（SSM）的新型视觉表示学习架构，旨在提高图像处理的效率和性能。Vim通过结合位置嵌入和双向SSM，能够有效地处理高分辨率图像，同时在ImageNet分类、COCO目标检测和ADE20K语义分割等任务上超越了现有的Vision Transformer模型。Vim在处理高分辨率图像时展现出了显著的计算和内存效率优势，例如在处理1248×1248分辨率图像时，比DeiT快2.8倍且节省了86.8%的GPU内存。

- 0️⃣2️⃣ **VMamba: Visual State Space Model** <br>
Paper Link: [📄📄📄](https://arxiv.org/pdf/2401.10166.pdf), Code：[✅✅✅](https://github.com/MzeroMiko/VMamba) <br>
Summary: 这篇论文提出了VMamba，一种新型的视觉状态空间模型（Visual State Space Model），它结合了全局感受野和动态权重，同时保持线性复杂度，以提高视觉表示学习的计算效率。VMamba使用应该CNN作为编码器，将图像转换为一维序列，然后使用应该Mamba模型作为解码器，将序列转换为所需的输出。为了解决视觉数据的方向敏感问题，论文引入了交叉扫描模块（Cross-Scan Module, CSM），以确保每个元素在特征图中整合来自不同方向的信息。实验结果表明，VMamba在图像分类、目标检测和语义分割等多种视觉任务上展现出了有希望的性能，并且在图像分辨率增加时，与现有基准相比显示出更明显的优势。

## ⭐️Image Segmentation Based on Mamba

## ⭐️Image or Video Generation Based on Mamba

## ⭐️Image Dehazing Based on Mamba

## ⭐️Point Cloud Processing Based on Mamba

## ⭐️Graph Network Based on Mamba

## ⭐️Other Applications Based on Mamba
