\# CIFAR-10 Image Classification: Architectural Benchmarks \& Transfer Learning



A systematic empirical study evaluating inductive biases and representation learning across four distinct architectures on the CIFAR-10 dataset—progressing from a naive baseline to an optimized, two-stage fine-tuned ResNet-50 pipeline reaching \*\*94.23% test accuracy\*\*.



\---



\## 📊 Benchmark Results



| Stage | Model Strategy | Architecture Highlights | Key Bottleneck / Mechanism | Test Accuracy |

| :--- | :--- | :--- | :--- | :---: |

| \*\*1\*\* | \*\*Baseline MLP\*\* | Flatten + 3 Dense Layers (256 $\\rightarrow$ 128 $\\rightarrow$ 64) | Total loss of 2D spatial hierarchy | \~48.00% |

| \*\*2\*\* | \*\*Custom CNN\*\* | 3x \[Conv2D + MaxPool + Dropout] + Dense | Constrained parameter capacity | \~65.20% |

| \*\*3\*\* | \*\*ResNet-50 (Heavy Head)\*\* | 8x UpSampling + ResNet Base + Dense(128, 64) | Parameter explosion \& gradient variance | 93.42% |

| \*\*4\*\* | \*\*ResNet-50 (Optimized Pipeline)\*\* | 224x224 Resize + Augmentation + GAP + 2-Stage Tuning | Native receptive field alignment + Warm-up | \*\*94.23%\*\* |



\---



\## 🧠 Key Engineering \& Scientific Insights



1\. \*\*Catastrophic Forgetting Mitigation:\*\* Unfreezing deep backbones prematurely destroys pre-trained ImageNet feature representations. A two-stage protocol is essential:

&#x20;  - \*\*Phase 1 (Warm-up):\*\* Freeze backbone (`trainable=False`), train custom classification head ($LR = 10^{-3}$).

&#x20;  - \*\*Phase 2 (Selective Fine-tuning):\*\* Unfreeze top 35 layers (layers 140+), fine-tune with conservative learning rate ($LR = 10^{-5}$) and early stopping.

2\. \*\*Global Average Pooling vs. Flattening:\*\* Replacing dense fully-connected layers with `GlobalAveragePooling2D` acts as an inductive regularizer, collapsing the spatial $7 \\times 7 \\times 2048$ tensor into a 2048-dim vector without parameter bloat.

3\. \*\*Resolution Alignment:\*\* Native ResNet-50 filters are tuned to $224 \\times 224$ receptive fields. Upscaling $32 \\times 32$ CIFAR-10 images preserves high-frequency semantic features.



\---



\## 🛠️ Tech Stack \& Implementation

\- \*\*Framework:\*\* Python, TensorFlow / Keras 3

\- \*\*Data Augmentation:\*\* Horizontal Flip, Random Translation

\- \*\*Optimization:\*\* Adam, EarlyStopping (`val\_loss` monitoring)



\---



\## 🔗 Links

\- \[Kaggle Notebook](https://www.kaggle.com/code/diamondtda/cifar-10-benchmarking-architectures-fine-tuning)
