# A-survey-on-Mamba
📊 **Summary Table**
| Field | Number of Papers | Remarks |
| ------- | :----: | ------: |
| Improvements and Optimizations Based on Mamba | 15 | \ |
| Vision Mamba | 3 | \  |
| Large Language Model Based on Mamba | 2 | \  |
| Image Segmentation Based on Mamba | 13 | \  |
| Target Detection Based on Mamba | 1 | \  |
| Multi-modal model Based on Mamba | 1 | \  |
| Image or Video Generation Based on Mamba | 4 | \  |
| Remote Sensing Images Based on Mamba Based on Mamba | 1 | \  |
| Image Dehazing Based on Mamba | 1 | \  |
| Point Cloud Processing Based on Mamba | 1 | \  |
| Graph Network Based on Mamba | 2 | \  |
| Other Applications Based on Mamba | 7 | \  |
| Total | 51 | \  |

## ☀️ Mamba
- **Linear-Time Sequence Modeling with Selective State Spaces** <br>
📆 2023.12, Paper Link: [📄📄📄](https://arxiv.org/abs/2312.00752), Code：[✅✅✅](https://github.com/state-spaces/mamba) <br>
📌 Notes：

- 0️⃣1️⃣ **Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers** <br>
📆 NeurIPS 2021, Paper Link: [📄📄📄](https://proceedings.neurips.cc/paper_files/paper/2021/hash/05546b0e38ab9175cd905eebcc6ebb76-Abstract.html), Code：❗️❗️❗️ <br>
📖 Summary: 这篇文章介绍了一种新的序列模型——线性状态空间层（LSSL），它结合了循环神经网络（RNNs）、卷积神经网络（CNNs）和神经微分方程（NDEs）的特点，旨在提高时间序列数据建模的效率和性能。LSSL通过模拟连续时间状态空间表示来处理序列数据，能够处理长序列依赖关系，并且在多个时间序列基准测试中取得了最先进的结果。此外，文章还提出了一种结构化矩阵A的可训练子集，赋予LSSL长期记忆能力，并通过理论和实证分析展示了其在处理非常长序列方面的潜力。<br>
📌 Notes：SSL（State-Space Layers）的首次提出

- 0️⃣2️⃣ **Efficiently Modeling Long Sequences with Structured State Spaces** <br>
📆 2021.11, Paper Link: [📄📄📄](https://arxiv.org/abs/2111.00396), Code：[✅✅✅](https://github.com/state-spaces/s4) <br>
📖 Summary: 这篇文章提出了一种新的序列模型——Sequences with Structured State Space (S4)，它基于状态空间模型（SSM），通过引入低秩修正和稳定的对角化技术，显著提高了处理长序列数据的效率。S4模型在多个基准测试中取得了先进的性能，特别是在处理长范围依赖性（LRDs）任务时，与现有的Transformer模型相比，显示出更快的训练速度和更低的内存使用。此外，文章还展示了S4在一系列广泛的任务中的潜力，包括图像分类、语言建模和时间序列预测，证明了其作为通用序列模型的潜力。<br>
📌 Notes：S4的首次提出，得到广泛关注的文章

- 0️⃣3️⃣ **HiPPO: Recurrent Memory with Optimal Polynomial Projections** <br>
📆 20220, Paper Link: [📄📄📄](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html), Code：[✅✅✅](https://github.com/HazyResearch/hippo-code) <br>
📖 Summary: 这篇文章介绍了HiPPO（High-order Polynomial Projection Operators）框架，这是一种用于在线压缩连续信号和离散时间序列的通用方法，通过将信号投影到多项式基上，并根据过去每个时间步的重要性来优化。HiPPO框架提供了一种理论上的优化解决方案，用于在线函数逼近问题，并且能够通过不同的度量方法来适应不同的时间尺度，从而避免了对时间尺度的先验假设。实验结果表明，HiPPO-LegS（一种特殊的HiPPO机制）在处理长期依赖关系的任务上，如排列MNIST数据集，达到了98.3%的准确率，并且在处理分布偏移和缺失数据的时序分类任务上，表现出比传统RNN和神经ODE基线更好的鲁棒性。<br>
📌 Notes：HiPPO的首次提出


## ⭐️ Improvements and Optimizations Based on Mamba
- 0️⃣1️⃣ **MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.04081), Code：❗️❗️❗️ <br>
📖 Summary: 这篇文章提出了MoE-Mamba，这是一个结合了Mamba状态空间模型（SSM）和专家混合（Mixture of Experts, MoE）机制的新型模型，旨在提高语言模型的扩展性和训练效率。MoE-Mamba在训练步骤上比原始的Mamba模型减少了2.35倍，同时保持了与Mamba相当的性能，特别是在处理长序列时。文章还探讨了不同的MoE集成方案，并对MoE-Mamba的不同设计选择、专家数量和模型大小进行了全面评估，证明了其在提高大型语言模型训练效率方面的潜力。<br>
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

- 0️⃣7️⃣ **Repeat After Me: Transformers are Better than State Space Models at Copying** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.01032), Code：❗️❗️❗️ <br>
📖 Summary: 这篇文章比较了Transformers和广义状态空间模型（Generalized State Space Models, GSSMs）在复制任务上的性能。研究表明，尽管GSSMs在推理时效率更高，但在需要从输入上下文中复制信息的任务上，变换器模型表现更佳。文章通过理论分析和实证实验，证明了变换器能够处理指数级长度的字符串复制，而GSSMs由于固定大小的潜在状态而受到限制。最后，文章通过预训练的大型语言模型评估，发现变换器在从上下文中复制和检索信息方面显著优于状态空间模型。<br>
📌 Notes：Transformers和SSMs的对比

- 0️⃣8️⃣ **The Hidden Attention of Mamba Models** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.01590), Code：[✅✅✅](https://github.com/Zyphra/BlackMamba) <br>
📖 Summary: 这篇文章探讨了Mamba模型中的隐式注意力机制，揭示了这种高效的选择性状态空间模型（SSM）如何通过内部的注意力机制与Transformer模型中的自注意力层相类似。研究表明，Mamba模型通过独特的数据控制线性操作符实现了隐式注意力，这为解释Mamba模型的内部工作机制提供了新的视角，并有助于开发用于解释性人工智能（Explainable Artificial Intelligence, XAI）的工具。文章还展示了Mamba模型在计算机视觉领域的注意力可视化和解释性评估，证明了其与Transformer模型相当的解释性能力。<br>
📌 Notes：Mamba的可解释性（explainability）分析,将Mamba的S6 Layers进行处理简化为QKH，再和Transformer的QKV联系起来，提供了比较Mamba和Transformer两者performance, fairness, robustness, and weaknesses的方法。<br>
对于分类任务：在Pertubation Tests中，Mamba对于正向扰动的适应性优于Transformer，但负向扰动的适应性不如Transformer，这个值得思考。。。？（正向扰动：在这种设置下，研究者从输入图像中逐步移除像素，这些像素根据它们对模型预测的相关性从最高到最低排序。理想情况下，随着最相关的像素被移除，模型的准确率应该显著下降。）<br>
对于分割任务：Mamba的pixel accuracy和mIoU高于Transformer，但mAP低于Transformer。

- 0️⃣9️⃣ **BlackMamba: Mixture of Experts for State-Space Models** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.01771), Code：[✅✅✅](https://github.com/AmeenAli/HiddenMambaAttn) <br>
📖 Summary: 这篇文章介绍了BlackMamba，这是一种新型的混合专家（Mixture of Experts, MoE）架构，结合了Mamba状态空间模型（SSM）以提高语言模型的性能。BlackMamba在保持线性时间和内存复杂度的同时，展示了与Mamba和Transformer基线模型相竞争的性能，并在推理和训练的浮点运算（FLOPs）上表现更优。文章还提到了BlackMamba模型的开源发布，包括340M/1.5B和630M/2.8B两个版本，以及在特定数据集上训练的结果。<br>
📌 Notes：

- 1️⃣0️⃣ **SiMBA: Simplified Mamba-based Architecture for Vision and Multivariate Time series** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.15360), Code：[✅✅✅](https://github.com/badripatro/Simba) <br>
📖 Summary: 文章提出了一种名为SiMBA的新型架构，结合了Mamba状态空间模型（SSM）和EinFFT（爱因斯坦快速傅里叶变换）技术，用于视觉和多变量时间序列数据的处理。SiMBA通过EinFFT进行高效的通道建模，并利用Mamba块进行序列建模，解决了Mamba在大规模网络中的稳定性问题，并在多个基准测试中达到了最先进的性能。该架构在ImageNet和多个时间序列数据集上进行了广泛的性能研究，证明了其在处理长序列和图像识别任务中的有效性，并在一些任务上建立了新的最先进的性能标准。<br>
📌 Notes：

- 1️⃣1️⃣ **VMRNN: Integrating Vision Mamba and LSTM for Efficient and Accurate Spatiotemporal Forecasting** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.16536), Code：[✅✅✅](https://github.com/yyyujintang/VMRNN-PyTorch) <br>
📖 Summary: 这篇文章提出了一种名为VMRNN的新型循环神经网络结构，它结合了视觉Mamba模块和长短期记忆（LSTM）网络，用于高效准确的时空预测任务。VMRNN通过整合视觉Mamba块的长序列建模能力和LSTM的时空动态捕捉能力，显著提升了在多种任务上的预测性能，同时保持了较小的模型规模。通过在多个数据集上的广泛评估，VMRNN在减少计算复杂性的同时，展现了与或超越现有最先进方法的预测准确性。<br>
📌 Notes：

- 1️⃣2️⃣ **STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.12418), Code：❗️❗️❗️ <br>
📖 Summary: 本文提出了一种名为STG-Mamba的新型空间-时间图（STG）学习方法，该方法通过选择性状态空间模型（SSSM）来有效捕捉STG网络的动态演变。STG-Mamba采用编码器-解码器架构，并引入了图选择性状态空间块（GS3B）和卡尔曼滤波图神经网络（KFGN），以实现对STG数据的高效建模和预测。实验结果表明，STG-Mamba在多个基准STG预测数据集上不仅超越了现有最先进方法的性能，还显著降低了计算成本，展现了其在大规模STG学习任务中的潜力。<br>
📌 Notes：

- 1️⃣3️⃣ **Theoretical Foundations of Deep Selective State-Space Models** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.19047), Code：❗️❗️❗️ <br>
📖 Summary: 这篇文章探讨了深度选择性状态空间模型（Deep Selective State-Space Models）的理论基础，特别是如何通过引入简单的输入控制转换（selectivity mechanism）来增强模型的表达能力。文章使用粗糙路径理论（Rough Path Theory）的工具，证明了当随机线性递归（random linear recurrences）配备这种机制时，隐藏状态能够捕捉到输入在不同时间尺度上的非线性交互。此外，文章还提出了一个分析框架，用于理解和比较未来的SSM变体，并指出了这些模型在处理序列数据时的高效性和准确性。<br>
📌 Notes：重点讨论selectivity mechanis的文章

- 1️⃣4️⃣ **Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.03900), Code：[✅✅✅](https://github.com/chengkai-liu/Mamba4Rec) <br>
📖 Summary: 这篇文章提出了Mamba4Rec，一个基于选择性状态空间模型（SSMs）的高效序列推荐系统，旨在解决Transformer模型在处理长序列时推理效率低下的问题。Mamba4Rec结合了Mamba模块和一系列序列建模技术，如位置嵌入、残差连接和层归一化，以提高模型性能并保持推理效率。实验结果表明，Mamba4Rec在两个公共数据集上的有效性和效率均超过了基于RNN和基于注意力的基线模型。<br>
📌 Notes：

- 1️⃣5️⃣ **MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.19888), Code：[✅✅✅](https://mambamixer.github.io) <br>
📖 Summary: 这篇文章介绍了MambaMixer，这是一种高效的选择性状态空间模型（SSM），它通过双重选择机制在标记（tokens）和通道（channels）上进行操作。MambaMixer通过选择性地混合和过滤信息，提高了长序列建模的性能，并在图像分类、目标检测、语义分割以及时间序列预测等任务中取得了与现有模型相媲美或更好的结果。此外，MambaMixer的设计允许模型直接访问早期特征，增强了信息流动，并通过数据依赖的权重提高了模型的泛化能力。<br>
📌 Notes：很完善的模型


## ⭐️ Vision Mamba
- 0️⃣1️⃣ **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.09417), Code：[✅✅✅](https://github.com/hustvl/Vim) <br>
📖 Summary: 这篇论文提出了Vision Mamba (Vim)，一种基于双向状态空间模型（SSM）的新型视觉表示学习架构，旨在提高图像处理的效率和性能。Vim通过结合位置嵌入和双向SSM，能够有效地处理高分辨率图像，同时在ImageNet分类、COCO目标检测和ADE20K语义分割等任务上超越了现有的Vision Transformer模型。Vim在处理高分辨率图像时展现出了显著的计算和内存效率优势，例如在处理1248×1248分辨率图像时，比DeiT快2.8倍且节省了86.8%的GPU内存。<br>
📌 Notes：

- 0️⃣2️⃣ **VMamba: Visual State Space Model** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.10166), Code：[✅✅✅](https://github.com/MzeroMiko/VMamba) <br>
📖 Summary: 这篇论文提出了VMamba，一种新型的视觉状态空间模型（Visual State Space Model），它结合了全局感受野和动态权重，同时保持线性复杂度，以提高视觉表示学习的计算效率。VMamba使用应该CNN作为编码器，将图像转换为一维序列，然后使用应该Mamba模型作为解码器，将序列转换为所需的输出。为了解决视觉数据的方向敏感问题，论文引入了交叉扫描模块（Cross-Scan Module, CSM），以确保每个元素在特征图中整合来自不同方向的信息。实验结果表明，VMamba在图像分类、目标检测和语义分割等多种视觉任务上展现出了有希望的性能，并且在图像分辨率增加时，与现有基准相比显示出更明显的优势。<br>
📌 Notes：

- 0️⃣3️⃣ **EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.09977), Code：[✅✅✅](https://github.com/TerryPei/EfficientVMamba) <br>
📖 Summary: 本文介绍了一种名为EfficientVMamba的轻量级视觉模型，该模型通过结合视觉状态空间模型（SSM）和空洞选择性扫描（ES2D）方法，实现了高效的全局和局部特征提取。EfficientVMamba通过在网络的早期阶段使用SSM块捕获全局信息，并在后续阶段采用倒置残差块（Inverted Residual blocks）提取局部特征，有效地平衡了模型的计算复杂度和性能。实验结果表明，EfficientVMamba在保持较低计算成本的同时，在图像分类、目标检测和语义分割等多个视觉任务上取得了与现有轻量级模型相当或更好的性能。<br>
📌 Notes：


## ⭐️ Large Language Model Based on Mamba
- 0️⃣1️⃣ **ClinicalMamba: A Generative Clinical Language Model on Longitudinal Clinical Notes** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.05795), Code：❗️❗️❗️ <br>
📖 Summary: 这篇文章介绍了ClinicalMamba，这是一个专门针对医疗领域设计的Mamba语言模型，它通过在大量纵向临床笔记上进行预训练，来解决医疗领域独特的语言特征和信息处理需求。ClinicalMamba模型在处理长文本时表现出色，与Mamba和临床Llama相比，它在少次学习情况下在速度和性能上都取得了显著的基准测试结果，超越了现有的临床语言模型和像GPT-4这样的大型语言模型。文章还讨论了ClinicalMamba在临床信息提取任务中的性能，特别是在队列选择和国际疾病分类（ICD）编码任务上，展示了其在处理长文本临床信息提取任务中的优越性。<br>
📌 Notes：

- 0️⃣2️⃣ **Long-Context Language Modeling with Parallel Context Encoding** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.16617), Code：[✅✅✅](https://github.com/princeton-nlp/CEPE) <br>
📖 Summary: 这篇文章介绍了一种名为Context Expansion with Parallel Encoding (CEPE)的框架，它通过添加小型编码器和交叉注意力模块来扩展大型语言模型（LLMs）的上下文窗口，从而有效处理更长的输入序列。CEPE在保持高效率和通用性的同时，通过在LLAMA-2模型上的应用展示了其在语言建模和上下文学习任务中的优秀性能，特别是在检索增强应用中，与其他长上下文模型相比，CEPE在检索增强语言建模和开放领域问答任务中表现出更好的性能。此外，文章还提出了CEPE的变体CEPE-DISTILLED (CEPED)，它可以通过未标记数据扩展指令调整模型的上下文窗口，进一步提高了长文本理解任务的性能。<br>
📌 Notes：


## ⭐️ Image Segmentation Based on Mamba
- 0️⃣1️⃣ **U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation** <br>
📆 2024.1, Paper Link: [📄📄📄](https://arxiv.org/abs/2401.04722), Code：[✅✅✅](https://github.com/bowang-lab/U-Mamba) <br>
📖 Summary: U-Mamba是一种新型的生物医学图像分割网络，它融合了CNN的局部特征提取和SSM的长距离依赖性建模能力，增强了图像的长距离依赖关系。该网络具有自适应不同数据集的自配置机制，无需手动调整。实验结果显示U-Mamba在多个生物医学图像分割任务上超越了现有的CNN和Transformer模型，展现了其作为一种高效、灵活的分割工具的潜力。<br>
📌 Notes：

- 0️⃣2️⃣ **SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation** <br>
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

- 1️⃣0️⃣ **Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.05079), Code：[✅✅✅](https://github.com/ziyangwang007/Mamba-UNet) <br>
📖 Summary: 这篇文章介绍了Mamba-UNet，这是一种基于纯视觉Mamba（VMamba）块的UNet风格网络，用于医学图像分割。Mamba-UNet结合了U-Net的编码器-解码器结构和Mamba的长序列处理能力，通过跳连接保留了不同尺度的空间信息，以捕捉医学图像中的复杂细节和更广泛的语义上下文。实验结果表明，Mamba-UNet在公开的MRI心脏多结构分割数据集上的表现优于传统的U-Net和Swin-UNet。作者提供了源代码，并计划将来在不同的医学图像分割任务和半监督学习中进一步探索Mamba-UNet的应用。<br>
📌 Notes：

- 1️⃣1️⃣ **VM-UNET-V2: Rethinking Vision Mamba UNet for Medical Image Segmentation** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.09157), Code：[✅✅✅](https://github.com/nobodyplayer1/VM-UNetV2) <br>
📖 Summary: 这篇文章介绍了一种名为VM-UNetV2的医学图像分割模型，该模型基于状态空间模型（SSM）和Vision Mamba UNet（VMamba）架构，旨在提高对长距离依赖的建模能力，同时保持线性计算复杂度。通过在多个公共数据集上的实验，VM-UNetV2展示了其在医学图像分割任务中的竞争力，特别是在处理胃肠病和皮肤病变图像时。此外，该研究还探讨了模型的不同配置和深度监督机制对分割性能的影响，并证明了VM-UNetV2在计算效率和准确性方面的优越性。<br>
📌 Notes：直接基于VMamba的应用已经来了

- 1️⃣2️⃣ **SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation** <br>
📆 2024.4, Paper Link: [📄📄📄](https://arxiv.org/abs/2404.01065), Code：[✅✅✅](https://github.com/isbrycee/T-Mamba) <br>
📖 Summary: 这篇文章提出了T-Mamba，一种用于三维锥形束计算机断层扫描（3D CBCT）图像中牙齿分割的深度学习模型，它通过集成共享位置编码和基于频率的特征到视觉状态空间模型（Vision Mamba）中来解决空间位置保持和频率域特征增强的局限性。T-Mamba是首个将基于频率的特征引入到视觉状态空间模型中的工作，它通过设计一个门控选择单元来适应性地整合空间域和频率域的特征。实验表明，T-Mamba在公开的牙齿CBCT数据集上取得了新的最先进结果，大幅超越了先前的最佳方法，例如IoU提高了3.63%，SO提高了2.43%，DSC提高了2.30%，HD降低了4.39mm，ASSD降低了0.37mm。<br>
📌 Notes：

- 1️⃣3️⃣ **Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model** <br>
📆 2024.4, Paper Link: [📄📄📄](https://arxiv.org/abs/2404.01705), Code：[✅✅✅](https://github.com/zhuqinfeng1999/Samba) <br>
📖 Summary: 这篇文章介绍了Samba，一种基于Mamba的状态空间模型（SSM），专为高分辨率遥感图像的语义分割设计。Samba采用编码器-解码器架构，利用Samba块作为编码器进行高效的多级语义信息提取，并使用UperNet作为解码器。在LoveDA数据集上的评估显示，Samba在性能上超越了当前最先进的CNN和ViT方法，为基于Mamba的遥感图像分割技术设定了新的基准。文章的主要贡献包括提出Samba架构、展示Mamba在遥感图像分割中的潜力，并为未来工作提供了洞见和潜在方向。<br>
📌 Notes：


## ⭐️ Multi-modal Model Based on Mamba
- 0️⃣1️⃣ **ReMamber: Referring Image Segmentation with Mamba Twister** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.17839), Code：❗️❗️❗️ <br>
📖 Summary: 这篇文章提出了MIM-ISTD（Mamba-in-Mamba for Efficient Infrared Small Target Detection），这是一种结合了Mamba状态空间模型的红外小目标检测方法，旨在提高检测的效率和准确性。MIM-ISTD通过将图像分割成“视觉句子”和“视觉单词”，并使用外部和内部Mamba块来分别提取全局和局部特征，从而有效地处理了红外图像中的小目标检测任务。实验结果表明，与现有的最先进方法相比，MIM-ISTD在保持高准确性的同时，显著提高了计算速度，降低了GPU内存使用，特别是在处理高分辨率红外图像时。<br>
📌 Notes：


## ⭐️ Target Detection Based on Mamba
- 0️⃣1️⃣ **MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.02148), Code：[✅✅✅](https://github.com/txchen-USTC/MiM-ISTD) <br>
📖 Summary: 文章提出了一种名为ReMamber的新型Referring Image Segmentation (RIS) 架构，该架构通过整合Mamba模型和多模态Mamba Twister模块来处理图像和文本之间的交互，实现了高效的特征融合和图像分割。ReMamber通过独特的通道和空间扭曲机制，显式地建模图像-文本交互，并通过融合文本和视觉特征来提高分割精度，在三个具有挑战性的基准测试中达到了最先进的性能。研究还深入分析了ReMamber架构，并讨论了使用Mamba进行其他融合设计的可能性，为未来多模态理解领域的研究提供了宝贵的视角和方向。<br>
📌 Notes：


## ⭐️ Image or Video Generation Based on Mamba
- 0️⃣1️⃣ **Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces** <br>
📆 2024.2, Paper Link: [📄📄📄](https://browse.arxiv.org/pdf/2402.00789), Code：[✅✅✅](https://github.com/bowang-lab/Graph-Mamba) <br>
📖 Summary: 这篇论文介绍了Graph-Mamba，这是一种新型图网络模型，它通过集成Mamba模块来增强图网络中的长距离上下文建模。Graph-Mamba利用选择性状态空间模型（SSM）来实现输入依赖的图稀疏化，并通过节点优先级和排列策略来提高预测性能。实验表明，Graph-Mamba在长距离图预测任务上超越了现有方法，并且在计算成本和GPU内存消耗方面都大幅减少。<br>
📌 Notes：

- 0️⃣2️⃣ **VideoMamba: State Space Model for Efficient Video Understanding** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2403.06977), Code：[✅✅✅](https://github.com/OpenGVLab/VideoMamba) <br>
📖 Summary: 这篇文章介绍了VideoMamba，这是一个基于状态空间模型（SSM）的视频理解模型，专为高效处理视频内容而设计。VideoMamba通过其线性复杂度操作符，能够有效处理长视频序列，并且在短视频和长期视频理解任务中展现出优越性能。文章通过广泛的评估展示了VideoMamba在视觉领域的可扩展性、对短期动作识别的敏感性、在长期视频理解中的优越性以及与其他模态的兼容性，并通过开源代码和模型来促进未来的研究工作。<br>
📌 Notes：

- 0️⃣3️⃣ **MambaMIR: An Arbitrary-Masked Mamba for Joint Medical Image Reconstruction and Uncertainty Estimation** <br>
📆 2024.3, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.18451), Code：❗️❗️❗️<br>
📖 Summary: 本文提出了MambaMIR，一种基于Mamba模型的医学图像重建方法，以及其基于生成对抗网络（GAN）的变体MambaMIR-GAN，用于联合医学图像重建和不确定性估计。MambaMIR通过引入任意掩蔽机制，有效地适应了图像重建任务，并为后续的蒙特卡洛基于不确定性估计提供了随机性。实验结果表明，MambaMIR和MambaMIR-GAN在多种医学图像重建任务上，包括快速MRI和SVCT，达到了与现有最先进方法相当或更优的重建结果，并提供了估计的不确定性图，进一步洞察重建质量的可靠性。<br>
📌 Notes：

- 0️⃣4️⃣ **SpikeMba: Multi-Modal Spiking Saliency Mamba for Temporal Video Grounding** <br>
📆 2024.4, Paper Link: [📄📄📄](https://arxiv.org/abs/2404.01174), Code：❗️❗️❗️<br>
📖 Summary: 这篇文章介绍了SpikeMba，一种用于视频内容理解中的时间视频定位（Temporal Video Grounding, TVG）的多模态脉冲神经网络（Spiking Neural Networks, SNNs）和状态空间模型（State Space Models, SSMs）的集成方法。SpikeMba通过引入相关的槽（slots）来增强模型的记忆力，并利用脉冲显著性检测器来准确定位视频中的显著时刻。实验表明，SpikeMba在主流基准测试中持续超越了最先进的方法。<br>
📌 Notes：视频理解模型


## ⭐️ Remote Sensing Images Based on Mamba
- 0️⃣1️⃣ **RSMamba: Remote Sensing Image Classification with State Space Model** <br>
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.04139), Code：[✅✅✅](https://github.com/KyanChen/RSMamba) <br>
📖 Summary: 这篇论文介绍了一种名为RSMamba的新型架构，用于遥感图像分类。RSMamba基于状态空间模型（SSM），并通过动态多路径激活机制，有效整合了全局感受野和线性建模复杂度的优势，以提高对遥感场景的分类精度。通过在多个遥感图像分类数据集上的实验，RSMamba展现了超越现有基于CNN和Transformer的分类方法的性能，显示出作为未来视觉基础模型骨干网络的潜力。<br>
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
📆 2024.2, Paper Link: [📄📄📄](https://arxiv.org/abs/2402.00789), Code：[✅✅✅](https://github.com/bowang-lab/Graph-Mamba) <br>
📖 Summary: 这篇文章提出了Graph-Mamba，这是一种新型的图网络模型，它通过将Mamba模块与输入依赖的节点选择机制相结合，来增强图网络中的长距离上下文建模。Graph-Mamba通过创新的图网络设计和状态空间模型（SSMs）的适应性改进，实现了对非序列图数据的高效处理。实验结果表明，Graph-Mamba在十个基准数据集上的长距离图预测任务中超越了现有方法，并且在计算成本（FLOPs）和GPU内存消耗方面都有显著减少，展现了其在大规模图数据上的高效性和优越性能。<br>
📌 Notes：

- 0️⃣2️⃣ **Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces** <br>
📆 2024.2, 📖 Summary: [📄📄📄](https://arxiv.org/abs/2402.12192), Code：[✅✅✅](https://github.com/bowang-lab/Graph-Mamba) <br>
Summary: 这篇论文提出了Pan-Mamba，一个新颖的全景锐化网络（Pan-sharpening），它利用状态空间模型（特别是Mamba模型）的高效全局信息建模能力来进行图像融合。Pan-Mamba通过定制的通道交换Mamba和跨模态Mamba模块，实现了跨模态信息的有效交换和融合，从而在多个数据集上超越了现有的全景锐化方法。这是首次尝试将Mamba模型应用于全景锐化技术，为该领域建立了新的前沿，并提供了源代码供公众访问。<br>
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

- 0️⃣3️⃣ **MambaTab: A Simple Yet Effective Approach for Handling Tabular Data** <br>
📆 2024.1, 📖 Summary: [📄📄📄](https://arxiv.org/abs/2401.08867), Code：❗️❗️❗️ <br>
Summary: 这篇文章介绍了MambaTab，这是一个基于结构化状态空间模型（SSM）的创新方法，用于处理表格数据。MambaTab利用Mamba，一种新兴的SSM变体，实现了端到端的监督学习，相较于现有的深度学习模型，它在参数数量上显著减少，同时几乎不需要数据预处理。实验结果表明，MambaTab在多个公共数据集上的性能超越了当前最先进的方法，同时在特征增量学习设置下表现出色，证明了其作为一种轻量级、即插即用的表格数据解决方案的潜力。<br>
📌 Notes：

- 0️⃣4️⃣ **Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data** <br>
📆 2024.2, 📖 Summary: [📄📄📄](https://arxiv.org/abs/2402.05892), Code：[✅✅✅](https://github.com/jacklishufan/Mamba-ND) <br>
Summary: 这篇文章介绍了Mamba-ND，这是一种将Mamba架构扩展到多维数据的新型设计，通过在不同层之间交替数据的展开顺序来处理图像、视频和科学数据等多维输入。Mamba-ND在保持线性复杂度的同时，在多个基准数据集上展示了与最先进模型相媲美的性能，并显著减少了参数数量。文章通过广泛的实验比较了Mamba-ND与其他替代方案的效果，并提供了代码供公众使用。<br>
📌 Notes：

- 0️⃣5️⃣ **Res-VMamba: Fine-Grained Food Category Visual Classification Using Selective State Space Models with Deep Residual Learning** <br>
📆 2024.2, 📖 Summary: [📄📄📄](https://arxiv.org/abs/2402.15761), Code：[✅✅✅](https://github.com/ChiShengChen/ResVMamba) <br>
Summary: 这篇文章提出了一种名为Res-VMamba的模型，用于细粒度食品类别的视觉分类，该模型结合了深度残差学习和选择性状态空间模型。通过使用新的数据集CNFOOD-241，研究结果表明Res-VMamba在没有预训练权重的情况下达到了79.54%的分类准确率，超越了当前的最先进模型。文章还讨论了食品图像识别的挑战，并强调了使用统一大小图像的数据集对于提高模型性能的重要性。<br>
📌 Notes：

- 0️⃣6️⃣ **MambaTalk: Efficient Holistic Gesture Synthesis with Selective State Space Models** <br>
📆 2024.3, 📖 Summary: [📄📄📄](https://arxiv.org/abs/2403.09471), Code：❗️❗️❗️ <br>
Summary: 这篇文章提出了MambaTalk，一个基于选择性状态空间模型（SSMs）的高效整体手势合成框架，通过结合离散运动先验和多模态集成来增强手势的多样性和节奏。MambaTalk通过两阶段建模策略，利用从VQ-VAE中提取的离散运动先验来解决直接应用选择性扫描机制导致的手势抖动问题，并通过多个Mamba模块来细化不同身体部位的潜在空间表示。实验表明，MambaTalk在BEATX标准数据集上的表现匹配或超过了当前最先进的模型，并且在训练速度上也显示出优越性。<br>
📌 Notes：

- 0️⃣7️⃣ **HARMamba: Efficient Wearable Sensor Human Activity Recognition Based on Bidirectional Selective SSM** <br>
📆 2024.3, 📖 Summary: [📄📄📄](https://arxiv.org/abs/2403.20183), Code：❗️❗️❗️ <br>
Summary: 这篇文章提出了HARMamba，一个基于双向选择性状态空间模型（SSM）的轻量级可穿戴传感器人体活动识别（HAR）框架，旨在解决实时活动识别场景中的计算资源限制问题。HARMamba通过将传感器数据分割成“patches”并独立学习每个通道，同时使用位置嵌入作为输入标记，通过分类头实现活动分类，与基于Transformer的模型相比，HARMamba在减少计算和内存开销的同时，实现了更优的性能。此外，HARMamba在四个公共活动数据集上进行了广泛的测试，包括PAMAP2、WISDM、UNIMIB和UCI，证明了其在活动识别任务中的卓越性能。<br>
📌 Notes：
