# List_of_Segmentation_Models

[참고]( https://github.com/mrgloom/awesome-semantic-segmentation )

1. FCN(Fully Convolutional Network)
2. Convolutional Models with Graphical Models
3. Encoder-Decoder Based
4. Multi-Scale and Pyramid Network Based
5. R-CNN Based*(for Instance Segmentation)*
6. Dilated Convolutional Models and DeepLab Family
7. RNN(Recurrent Neural Network)
8. Attention-Based
9. **Ganerative Models and Adversarial Training**
10. **CNN Models with Active Contour Models**
11. Others

## 1. Semantic Segmentation

|      | Models                                | Type                                            | Code                                                         | Paper                                                        |
| ---- | ------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | FCN (VGG-16)                          | FCN                                             | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) |
| 2    | CNN+CRF                               | Convolutional Models with Graphical Models      |                                                              | [link](https://arxiv.org/abs/1502.03240)                     |
| 3    | ParseNet                              | FCN + global context                            | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/abs/1506.04579)                     |
| 4    | Dilated Convolutional Net             | Dilated Convolutional Models and DeepLab Family | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link]()                                                     |
| 5    | Deconvolution Network(DeconvNet)      | Encoder-Decoder                                 | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cheng_Locality-Sensitive_Deconvolution_Networks_CVPR_2017_paper.pdf) |
| 6    | ReSeg                                 | RNN                                             | [PyTorch](https://github.com/Wizaron/reseg-pytorch)          | [link](https://re.public.polimi.it/retrieve/handle/11311/997624/138323/4.ReSeg.pdf) |
| 7    | U-Net                                 | Encoder-Decoder                                 | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/pdf/1505.04597.pdf)                 |
| 8    | Pyramid Scene Parsing Network(PSPNet) | Multi-Scale and Pyramid Network                 | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/pdf/1612.01105.pdf)                 |
| 9    | ENet                                  | Dilated Convolutional Models and DeepLab Family | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/pdf/1606.02147.pdf)                 |
| 10   | SegNet                                | Encoder-Decoder                                 | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/pdf/1511.00561.pdf)                 |
| 11   | RefineNet                             | Others                                          | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/pdf/1611.06612.pdf)                 |
| 12   | V-Net                                 | Encoder-Decoder                                 | [PyTorch](https://github.com/mattmacy/vnet.pytorch)          | [link](https://arxiv.org/abs/1606.04797)                     |
| 13   | DeepLab                               | Dilated Convolutional Models                    | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/pdf/1606.00915.pdf)                 |
| 14   | Feature Pyramid Network (FPN)         | Multi-Scale and Pyramid Network                 | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf) |
| 15   | DeepLab V3                            | Dilated Convolutional Models and DeepLab Family | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/pdf/1706.05587.pdf)                 |
| 16   | Global Convolutional Net(GCN)         | Others                                          | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/pdf/1703.02719.pdf)                 |
| 17   | FC-DenseNet(DenseNet)                 | Others                                          | tensorflow,keras,                                            | [link](https://arxiv.org/pdf/1611.09326.pdf)                 |
| 18   | Context Encoding Net(EncNet)          | Others                                          | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/abs/1803.08904)                     |
| 19   | Discriminative Feature Network(DFN)   | Attention-Based                                 | [PyTorch](https://github.com/ycszen/TorchSeg)                | [link](https://arxiv.org/abs/1804.09337)                     |
| 20   | Exfuse                                | ?                                               | ?                                                            | [link](https://arxiv.org/abs/1804.03821)                     |
| 21   | Dense-ASSP                            | Dilated Convolutional Models and DeepLab Family | [PyTorch](https://github.com/donnyyou/torchcv)               | [link](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf) |
| 22   | Dynamic Multi-Scale Filters(DM-Net)   | Multi-Scale and Pyramid Network                 |                                                              | [link](http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf) |
| 23   | FastFCN                               | FCN                                             | [PyTorch](https://github.com/wuhuikai/FastFCN)               | [link](https://arxiv.org/pdf/1903.11816.pdf)                 |
| 24   | Dual Attention Network(DANet)         | Attention-Based                                 | [Awsome](https://github.com/nerox8664/awesome-computer-vision-models) | [link](https://arxiv.org/pdf/1809.02983.pdf)                 |
| 25   | CC-Net: Criss-Cross Attention         | Attention-Based                                 | [PyTorch](https://github.com/speedinghzl/CCNet)              | [link](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_CCNet_Criss-Cross_Attention_for_Semantic_Segmentation_ICCV_2019_paper.pdf) |
| 26   | HRNet                                 | Encoder-Decoder                                 | [PyTorch](https://github.com/HRNet/HRNet-Semantic-Segmentation) | [link](https://arxiv.org/pdf/1904.04514.pdf)                 |



## 2. Instance Segmentation

|      | Models                      | Type  | Code                                                         | Paper                                        |
| ---- | --------------------------- | ----- | ------------------------------------------------------------ | -------------------------------------------- |
| 1    | DeepMask                    | R-CNN | [PyTorch](https://github.com/facebookresearch/deepmask)      | [link](https://arxiv.org/abs/1506.06204)     |
| 2    | Instance-Aware Segmentation | R-CNN |                                                              | [link](https://arxiv.org/abs/1512.04412)     |
| 3    | Mask-RCNN                   | R-CNN | [PyTorch](https://github.com/facebookresearch/maskrcnn-benchmark) | [link](https://arxiv.org/pdf/1703.06870.pdf) |
| 4    | Path Aggregation Network    | R-CNN | [PyTorch](https://arxiv.org/pdf/1803.01534.pdf)              | [link](https://arxiv.org/pdf/1803.01534.pdf) |
| 5    | Mask-Lab                    | R-CNN |                                                              | [link](https://arxiv.org/abs/1712.04837)     |
| 6    | TensorMask                  | R-CNN |                                                              | [link](https://arxiv.org/abs/1903.12174)     |

