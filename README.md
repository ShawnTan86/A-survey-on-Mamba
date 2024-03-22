# A-survey-on-Mamba
📊 **Summary Table**
| Field | Number of Papers | Remarks |
| ------- | :----: | ------: |
| Improvements and Optimizations Based on Mamba | 6 | \ |
| Vision Mamba | 2 | \  |
| Image Segmentation Based on Mamba | 9 | \  |
| Image or Video Generation Based on Mamba | 1 | \  |
| Image Dehazing Based on Mamba | 1 | \  |
| Point Cloud Processing Based on Mamba | 1 | \  |
| Graph Network Based on Mamba | 1 | \  |
| Other Applications Based on Mamba | 2 | \  |
| Total | 23 | \  |

## ☀️ Mamba
- **Linear-Time Sequence Modeling with Selective State Spaces** <br>
📆 2023.12, Paper Link: [📄📄📄](https://arxiv.org/abs/2312.00752), Code：[✅✅✅](https://github.com/state-spaces/mamba) <br>
📌 Notes：


## ⭐️ Improvements and Optimizations Based on Mamba
- 0️⃣1️⃣ **MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.04081), Code：❗️❗️❗️ <br>
📖 Summary: 这篇论文介绍了MambaByte，这是一种无需标记的、基于状态空间模型（SSM）的字节级语言模型，它通过自回归方式训练于字节序列上。MambaByte直接使用字节作为序列的基本单元。MambaByte在多个数据集上展示了与其他字节级模型相比的优越性能，并与最先进的基于子词的Transformer模型竞争，同时在推理速度上由于其线性扩展特性而受益。研究表明，MambaByte是一种有效的无需标记的语言建模方法，为未来的大型模型提供了一种可行的无需标记的语言建模可能性。<br>
📌 Notes：

- 0️⃣2️⃣ **MambaByte: Token-free Selective State Space Model** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.13660), Code：[✅✅✅](https://github.com/lucidrains/MEGABYTE-pytorch) <br>
📖 Summary: 这篇论文介绍了MambaByte，这是一种无需标记的、基于状态空间模型（SSM）的字节级语言模型，它通过自回归方式训练于字节序列上。MambaByte直接使用字节作为序列的基本单元。MambaByte在多个数据集上展示了与其他字节级模型相比的优越性能，并与最先进的基于子词的Transformer模型竞争，同时在推理速度上由于其线性扩展特性而受益。研究表明，MambaByte是一种有效的无需标记的语言建模方法，为未来的大型模型提供了一种可行的无需标记的语言建模可能性。<br>
📌 Notes：

- 0️⃣3️⃣ **LOCOST: State-Space Models for Long Document Abstractive Summarization** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.17919), Code：[✅✅✅](https://github.com/flbbb/locost-summarization) <br>
📖 Summary: 这篇论文提出了LOCOST，一种基于状态空间模型（SSMs）的编码器-解码器架构，LOCOST用一个Mamba模型作为编码器，将长文转换为一维序列，再使用一个Transformer模型作为解码器，用于处理长文本摘要任务。LOCOST的计算复杂度为O(L log L)，相较于基于稀疏注意力模式的现有模型，能够处理更长的序列，同时在训练和推理过程中节省大量内存。实验表明，LOCOST在长文档摘要任务上的性能与最先进的稀疏变换器相当，同时在处理超过600K令牌的输入时，取得了新的最先进结果，为长文本处理开辟了新的可能性。<br>
📌 Notes：

- 0️⃣4️⃣ **Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.05892), Code：❗️❗️❗️ <br>
📖 Summary: 这篇论文介绍了Mamba-ND，这是一种针对多维数据（如图像、视频等）的新型神经网络架构。Mamba-ND基于状态空间模型（SSM），通过在不同维度上交替处理输入数据，实现了与Transformer模型相当的性能，同时显著降低了参数数量和计算复杂度。在多个基准测试中，包括ImageNet-1K分类、HMDB-51和UCF-101动作识别、ERA5天气预测和BTCV 3D分割，Mamba-ND展现了与最先进模型相竞争的性能，并保持了线性复杂度。<br>
📌 Notes：

- 0️⃣5️⃣ **Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.10211), Code：[✅✅✅](https://github.com/raunaqbhirangi/hiss/tree/main) <br>
📖 Summary: 这篇论文介绍了一种新的连续序列到序列建模技术，名为层次化状态空间模型（HiSS），它通过在不同时间分辨率上堆叠结构化的状态空间模型来创建时间层次结构。HiSS在六个真实世界的传感器数据集上的表现超越了现有的最先进序列模型，如因果Transformer、LSTM、S4和Mamba，至少在均方误差（MSE）上提高了23%。此外，实验表明HiSS在小数据集上具有高效的扩展性，并且与现有的数据过滤技术兼容。论文还发布了CSP-Bench，这是一个公共的连续序列预测基准，包含六个真实世界标记数据集，旨在支持多样化的感官数据分析。<br>
📌 Notes：

- 0️⃣6️⃣ **Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.04248), Code：❗️❗️❗️ <br>
📖 Summary: 这篇论文介绍了一种新的连续序列到序列建模技术，名为层次化状态空间模型（HiSS），它通过在不同时间分辨率上堆叠结构化的状态空间模型来创建时间层次结构。HiSS在六个真实世界的传感器数据集上的表现超越了现有的最先进序列模型，如因果Transformer、LSTM、S4和Mamba，至少在均方误差（MSE）上提高了23%。此外，实验表明HiSS在小数据集上具有高效的扩展性，并且与现有的数据过滤技术兼容。论文还发布了CSP-Bench，这是一个公共的连续序列预测基准，包含六个真实世界标记数据集，旨在支持多样化的感官数据分析。<br>
📌 Notes：

## ⭐️ Vision Mamba
- 0️⃣1️⃣ **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.09417), Code：[✅✅✅](https://github.com/hustvl/Vim) <br>
📖 Summary: 这篇论文提出了Vision Mamba (Vim)，一种基于双向状态空间模型（SSM）的新型视觉表示学习架构，旨在提高图像处理的效率和性能。Vim通过结合位置嵌入和双向SSM，能够有效地处理高分辨率图像，同时在ImageNet分类、COCO目标检测和ADE20K语义分割等任务上超越了现有的Vision Transformer模型。Vim在处理高分辨率图像时展现出了显著的计算和内存效率优势，例如在处理1248×1248分辨率图像时，比DeiT快2.8倍且节省了86.8%的GPU内存。<br>
📌 Notes：

- 0️⃣2️⃣ **VMamba: Visual State Space Model** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.10166), Code：[✅✅✅](https://github.com/MzeroMiko/VMamba) <br>
📖 Summary: 这篇论文提出了VMamba，一种新型的视觉状态空间模型（Visual State Space Model），它结合了全局感受野和动态权重，同时保持线性复杂度，以提高视觉表示学习的计算效率。VMamba使用应该CNN作为编码器，将图像转换为一维序列，然后使用应该Mamba模型作为解码器，将序列转换为所需的输出。为了解决视觉数据的方向敏感问题，论文引入了交叉扫描模块（Cross-Scan Module, CSM），以确保每个元素在特征图中整合来自不同方向的信息。实验结果表明，VMamba在图像分类、目标检测和语义分割等多种视觉任务上展现出了有希望的性能，并且在图像分辨率增加时，与现有基准相比显示出更明显的优势。<br>
📌 Notes：

## ⭐️ Image Segmentation Based on Mamba
- 0️⃣1️⃣ **U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.04722), Code：[✅✅✅](https://github.com/bowang-lab/U-Mamba) <br>
📖 Summary: U-Mamba是一种新型的生物医学图像分割网络，它融合了CNN的局部特征提取和SSM的长距离依赖性建模能力，增强了图像的长距离依赖关系。该网络具有自适应不同数据集的自配置机制，无需手动调整。实验结果显示U-Mamba在多个生物医学图像分割任务上超越了现有的CNN和Transformer模型，展现了其作为一种高效、灵活的分割工具的潜力。<br>
📌 Notes：

- 0️⃣2️⃣ **Segmentation SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.13560), Code：[✅✅✅](https://github.com/ge-xing/SegMamba) <br>
📖 Summary: 该论文介绍了SegMamba，这是一种基于Mamba的新型3D医学图像分割模型，旨在有效捕捉整个体积特征的长距离依赖关系。SegMamba通过设计三向Mamba（ToM）模块和门控空间卷积（GSC）模块，提高了对3D特征的序列建模能力，SegMamba可以有效的捕捉三维图形长距离依赖关系，并在保持高效率的同时，展示了在多个数据集上的优越性能。此外，作者还提出了一个新的大规模数据集CRC-500，用于3D结直肠癌分割研究，并通过实验验证了SegMamba与传统CNN和基于Transformer的方法相比，在建模体积数据中的长距离依赖关系方面具有显著的推理效率和有效性。<br>
📌 Notes：

- 0️⃣3️⃣ **Vivim: a Video Vision Mamba for Medical Video Object Segmentation** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.14168), Code：[✅✅✅](https://github.com/scott-yjyang/Vivim) <br>
📖 Summary: 该论文提出了Vivim，这是一个基于视频视觉的Mamba框架，专门用于医学视频对象分割任务。Vivim通过设计的Temporal Mamba Block有效地压缩长期时空表示，并引入了边界感知约束来增强对医学图像中模糊病变的区分能力。Vivim可以有效地处理视频中动态变化和遮挡问题。实验结果表明，Vivim在甲状腺超声视频分割和结肠镜视频多息肉分割任务上优于现有方法，展示了其有效性和效率。<br>
📌 Notes：

- 0️⃣4️⃣ **VM-UNet: Vision Mamba UNet for Medical Image Segmentation** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.02491), Code：[✅✅✅](https://github.com/JCruan519/VM-UNet) <br>
📖 Summary: 这篇论文提出了VM-UNet，这是一个基于状态空间模型（SSMs）的U形网络架构，用于医学图像分割任务。VM-UNet利用视觉状态空间（VSS）块作为基础模块来捕获广泛的上下文信息，并构建了不对称的编码器-解码器结构。在ISIC17、ISIC18和Synapse数据集上的实验结果表明，VM-UNet在医学图像分割任务中表现出竞争力，这是首个基于纯SSM模型的医学图像分割模型。作者旨在建立一个基准，并为未来更高效、有效的SSM-based分割系统的发展提供有价值的见解。<br>
📌 Notes：

- 0️⃣5️⃣ **Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.03302), Code：[✅✅✅](https://github.com/JiarunLiu/Swin-UMamba) <br>
📖 Summary: 这篇论文提出了Swin-UMamba，这是一个基于Mamba模型的UNet架构，专为医学图像分割任务设计，并通过ImageNet预训练来提升性能。Swin-UMamba结合了预训练视觉模型的优势和为医学图像分割任务特别设计的解码器，实验结果显示其在多个医学图像数据集上的性能超越了CNN、ViT以及其他最新的Mamba模型。文章强调了ImageNet预训练在提高Mamba模型在医学图像分割任务中性能的重要性，并且在GitHub上公开了代码和模型。<br>
📌 Notes：

- 0️⃣6️⃣ **nnMamba: 3D Biomedical Image Segmentation, Classification and Landmark Detection with State Space Model** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.03526), Code：[✅✅✅](https://github.com/lhaof/nnMamba) <br>
📖 Summary: 这篇论文介绍了nnMamba，这是一个结合了卷积神经网络（CNNs）和状态空间序列模型（SSMs）的新型架构，专为3D生物医学图像分析任务设计，包括分割、分类和地标检测。nnMamba通过提出Mamba-InConvolution与通道-空间孪生学习（MICCSS）模块来模拟体素之间的长距离关系，并且在密集预测和分类任务中设计了通道缩放和通道顺序学习方法。在六个数据集上的广泛实验表明，nnMamba在一系列具有挑战性的任务中优于现有方法，证明了其在医学图像分析中长距离依赖建模的新标准。<br>
📌 Notes：

- 0️⃣7️⃣ **Semi-Mamba-UNet: Pixel-Level Contrastive Cross-Supervised Visual Mamba-based UNet for Semi-Supervised Medical Image Segmentation** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.07245), Code：[✅✅✅](https://github.com/ziyangwang007/Mamba-UNet) <br>
📖 Summary: 这篇论文提出了Semi-Mamba-UNet，这是一个结合了Visual Mamba和传统UNet架构的半监督学习（SSL）框架，用于医学图像分割。该框架通过双网络结构进行像素级对比学习和交叉监督学习，以提高在标注数据有限的情况下的特征学习能力。在公开的MRI心脏分割数据集上的评估显示，Semi-Mamba-UNet在多种SSL框架中表现出色，提供了优于现有技术的分割性能。论文还提供了源代码，以便公众访问和使用。<br>
📌 Notes：

- 0️⃣8️⃣ **P-Mamba: Marrying Perona Malik Diffusion with Mamba for Efficient Pediatric Echocardiographic Left Ventricular Segmentation** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.08506), Code：❗️❗️❗️ <br>
📖 Summary: 这篇论文提出了P-Mamba，这是一个结合了Perona-Malik扩散（PMD）和Mamba架构的新型网络，用于高效的儿童心脏超声心动图左心室分割。P-Mamba利用PMD分支进行噪声抑制和局部特征提取，同时使用Vision Mamba编码器分支来捕捉全局依赖性，以提高模型的准确性和效率。实验结果表明，P-Mamba在处理带有噪声的儿童超声心动图数据集上取得了优于现有模型的性能，展示了其在儿科心脏成像领域的潜力。<br>
📌 Notes：

- 0️⃣9️⃣ **Weak-Mamba-UNet: Visual Mamba Makes CNN and ViT Work Better for Scribble-based Medical Image Segmentation** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.10887), Code：[✅✅✅](https://github.com/ziyangwang007/Mamba-UNet) <br>
📖 Summary: 这篇论文提出了Weak-Mamba-UNet，一个用于医学图像分割的弱监督学习（WSL）框架，它结合了卷积神经网络（CNN）、视觉变换器（ViT）和最新的视觉Mamba（VMamba）架构，特别是针对基于涂鸦注释的数据。该框架的核心是一个协作和交叉监督机制，使用伪标签来促进网络间的迭代学习和精细化。在公开可用的MRI心脏分割数据集上进行的实验表明，Weak-Mamba-UNet在处理稀疏或不精确注释的情况下，其性能超过了仅使用UNet或SwinUNet的类似WSL框架。论文还提供了源代码的公开访问链接。<br>
📌 Notes：

## ⭐️ Image or Video Generation Based on Mamba
- 0️⃣1️⃣ **Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces** <br>
📆 2024.2, Paper Link: [📄📄📄](https://browse.arxiv.org/pdf/2402.00789), Code：[✅✅✅](https://github.com/bowang-lab/Graph-Mamba) <br>
📖 Summary: 这篇论文介绍了Graph-Mamba，这是一种新型图网络模型，它通过集成Mamba模块来增强图网络中的长距离上下文建模。Graph-Mamba利用选择性状态空间模型（SSM）来实现输入依赖的图稀疏化，并通过节点优先级和排列策略来提高预测性能。实验表明，Graph-Mamba在长距离图预测任务上超越了现有方法，并且在计算成本和GPU内存消耗方面都大幅减少。<br>
📌 Notes：

## ⭐️ Image Dehazing Based on Mamba
- 0️⃣1️⃣ **U-shaped Vision Mamba for Single Image Dehazing** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.04139), Code：[✅✅✅](https://github.com/zzr-idam) <br>
📖 Summary: 这篇论文提出了UVM-Net，一个基于U形网络结构的高效单图像去雾网络，它结合了卷积层的局部特征提取能力和状态空间序列模型（SSMs）捕获长距离依赖的能力。UVM-Net通过设计Bi-SSM模块来充分利用SSM的长距离建模能力，并且在多个公开数据集上的实验结果证明了其在图像去雾任务中的有效性。该方法为图像去雾以及其他图像恢复任务提供了一种高效的长距离依赖建模思路。<br>
📌 Notes：

## ⭐️ Point Cloud Processing Based on Mamba
- 0️⃣1️⃣ **PointMamba: A Simple State Space Model for Point Cloud Analysis** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.10739), Code：[✅✅✅](https://github.com/LMD0311/PointMamba) <br>
📖 Summary: 这篇论文提出了PointMamba，一个用于点云分析的简单状态空间模型（SSM），它具有全局建模能力和线性复杂度。PointMamba通过嵌入点云块作为输入，并采用重排序策略来增强SSM的全局建模能力，通过提供更合理的几何扫描顺序。实验结果表明，PointMamba在不同的点云分析数据集上超越了基于Transformer的模型，同时显著减少了约44.3%的参数和25%的浮点运算（FLOPs），展示了在构建基础3D视觉模型方面的潜力。作者希望PointMamba能为点云分析提供新的视角，并已在GitHub上提供了代码。<br>
📌 Notes：

## ⭐️ Graph Network Based on Mamba
- 0️⃣1️⃣ **Graph Mamba: Towards Learning on Graphs with State Space Models** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.08678), Code：❗️❗️❗️ <br>
📖 Summary: 这篇论文介绍了Graph Mamba Networks（GMNs），这是一种新的图神经网络（GNN）框架，基于状态空间模型（SSMs）来捕捉图数据中的长距离依赖关系。GMNs通过结合随机游走采样、双向选择性状态空间模型编码器、局部编码以及可选的结构/位置编码，提供了一种灵活、高效且可扩展的方法来处理图数据。实验结果表明，GMNs在处理长距离、小规模、大规模和异构图数据集上的基准测试中表现出色，同时在内存效率上超越了现有的图变换器（GTs）方法。<br>
📌 Notes：

## ⭐️ Other Applications Based on Mamba
- 0️⃣1️⃣ **FD-Vision Mamba for Endoscopic Exposure Correction** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.06378), Code：❗️❗️❗️ <br>
📖 Summary: 这篇论文介绍了FDVision Mamba（FDVM-Net），这是一个基于频率域的网络，用于内窥镜图像的曝光校正。该网络通过重建内窥镜图像的频率域来实现高质量的图像曝光校正。论文提出了一个双路径网络结构，使用C-SSM块作为基本功能单元，分别处理图像的相位和幅度信息，并通过新颖的频域交叉注意力机制来增强模型性能。实验结果表明，FDVM-Net在速度和准确性方面达到了最先进的结果，并且能够增强任意分辨率的内窥镜图像。<br>
📌 Notes：

- 0️⃣2️⃣ **Pan-Mamba: Effective pan-sharpening with State Space Model** <br>
📆 2024.2, 📖 Summary: [📄📄📄](https://arxiv.org/abs/2402.12192), Code：[✅✅✅](https://github.com/alexhe101/Pan-Mamba) <br>
Summary: 这篇论文提出了Pan-Mamba，一个新颖的全景锐化网络（Pan-sharpening），它利用状态空间模型（特别是Mamba模型）的高效全局信息建模能力来进行图像融合。Pan-Mamba通过定制的通道交换Mamba和跨模态Mamba模块，实现了跨模态信息的有效交换和融合，从而在多个数据集上超越了现有的全景锐化方法。这是首次尝试将Mamba模型应用于全景锐化技术，为该领域建立了新的前沿，并提供了源代码供公众访问。<br>
📌 Notes：

