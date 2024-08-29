# LFNet
This repository is the work of "LFNet: Lightweight Network for Local-Global Feature Fusion" based on pytorch implementation. We will continue to improve the relevant content.<br>
## Framework Overview
@import "https://github.com/lcxddn/LFNet/blob/main/Img/LFNet.pdf"
## Experimental results
### Table 1 Comparison of parameters and the FLOPs for each classified network.<br>
| Models | Flops(G) | Params(M) |
| ------------- | ------------- | ------------- |
| ResNet50  | 4.13  |  25.56  |
| ConvNeXt  | 4.46  | 27.8  |
| SwinTransformer  | 4.37  |  28.27  |
| PoolFormer  | 1.81  |  11.89  |
| CMT  | 1.22  |  9.44  |
| CoAtNet  | 3.35  |  17.76  |
| EMTCAL  | 4.23  |  27.31  |
| GCSANet   | 2.90  |  8.11  |
| DBGANet   | 13.21  |  108.37  |
| EfficientNetV2  | 2.90  |  21.46  |
| EdgeViTs  | 1.90  |  13.11  |
| RepViT  | 1.38  |  8.29  |
| MobileNetV3  | 0.23  |  5.48  |
| ShuffleNetV2   | 0.59  |  7.39  |
| LFNet(ours)  | 0.54  |  0.66  |

### Table 2 The ablation study of model validity by different modules.L. represents the local feature extraction method, G. represents the global feature extraction method, and the $\rightarrow$ indicates the sequence.
| Model  | L. $\rightarrow$ G. | G. $\rightarrow$ L. | G. | L. | Fusion | Acc(\%) |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| CGLFFNet  | |$\surd$| Content Cell  | Content Cell  | $\surd$  | 97.5  |
| LFENet  | | | |$\surd$|$\surd$| 97.14  |
| GLENet  | | |$\surd$| |$\surd$| 96.25  |
| LGFNet  |$\surd$ |   |   |   |   | 96.96  |
| LFNet  |$\surd$ |   |   |   |$\surd$ | 97.68  |

### Table 3 Effectiveness of different feature extraction methods.
| Model  | Conv-Module | Attention-Module | Acc(\%) |
| ------------- | ------------- | ------------- | ------------- |
| LFNet-Conv  | 2D-Conv  | LinerAttention  | 96.43  |
| LFNet-Atte  | ADC-Conv  | Self-Attention  | 96.60  |
| LFNet  | ADC-Conv  | LinerAttention  | 97.68  |

## Visualization
### Figure 1 Confusion matrix in the RSSCN7 dataset.
![image](https://github.com/lcxddn/LFNet/blob/main/Img/matrix_RSSCN7.pdf)

### Figure 2 Confusion matrix in the AID dataset.
![image](https://github.com/lcxddn/LFNet/blob/main/Img/matrix_AID.jpg)

### Figure 3 Heat map visualization.
![image](https://github.com/lcxddn/LFNet/blob/main/Img/heatmap.pdf)

