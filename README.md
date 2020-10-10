Acknowledgements ：   https://github.com/ChaofWang/Awesome-Super-Resolution

https://github.com/Kobaayyy/Awesome-CVPR2020-Low-Level-Vision

https://github.com/Kobaayyy/Awesome-ECCV2020-Low-Level-Vision


大佬们的github主页：https://github.com/cszn
# Quick navigation

- [repositories](#repositories)
  - [Awesome paper list](#Awesome-paper-list)
  - [Awesome repos](#Awesome-repos)
- [Datasets](#Datasets)
  - [Dataset collections](#Dataset-collections)
- [paper](#paper)
  - [Non-DL based approach](#Non-DL-based-approach)
  - [DL based approach](#DL-based-approach)
    - [2014-2016](#2014-2016)
    - [2017](#2017)
    - [2018](#2018)
    - [2019](#2019)
    - [2020](#2020)
- [Super Resolution workshop papers](#Super-Resolution-workshop-papers)
  - [NTIRE17](#NTIRE17-[papers](http://openaccess.thecvf.com/CVPR2017_workshops/CVPR2017_W12.py))
  - [NTIRE18](#NTIRE18-[papers](http://openaccess.thecvf.com/CVPR2018_workshops/CVPR2018_W13.py))
  - [PIRM18](#PIRM18-[Web](https://pirm2018.org/))
  - [NTIRE19](#NTIRE19-[papers](http://openaccess.thecvf.com/CVPR2019_workshops/CVPR2019_NTIRE.py))
  - [AIM19](#AIM19-[papers](http://openaccess.thecvf.com/ICCV2019_workshops/ICCV2019_AIM.py))
- [Super Resolution survey](#Super-Resolution-survey)

# Awesome-Super-Resolution（in progress）

Collect some super-resolution related papers, data and repositories.



## repositories

#### Awesome paper list:

[Single-Image-Super-Resolution](https://github.com/YapengTian/Single-Image-Super-Resolution)

[Super-Resolution.Benckmark](https://github.com/huangzehao/Super-Resolution.Benckmark)

[Video-Super-Resolution](https://github.com/flyywh/Video-Super-Resolution)

[VideoSuperResolution](https://github.com/LoSealL/VideoSuperResolution)

[Awesome Super-Resolution](https://github.com/ptkin/Awesome-Super-Resolution)

[Awesome-LF-Image-SR](https://github.com/YingqianWang/Awesome-LF-Image-SR)

[Awesome-Stereo-Image-SR](https://github.com/YingqianWang/Awesome-Stereo-Image-SR)

[AI-video-enhance](https://github.com/jlygit/AI-video-enhance)


#### Awesome repos:

|                             repo                             | Framework  |
| :----------------------------------------------------------: | :--------: |
| [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch)  |  PyTorch   |
| [Image-Super-Resolution](https://github.com/titu1994/Image-Super-Resolution) |   Keras    |
| [image-super-resolution](https://github.com/idealo/image-super-resolution) |   Keras    |
| [Super-Resolution-Zoo](https://github.com/WolframRhodium/Super-Resolution-Zoo) |   MxNet    |
| [super-resolution](https://github.com/krasserm/super-resolution) |   Keras    |
|  [neural-enhance](https://github.com/alexjc/neural-enhance)  |   Theano   |
|          [srez](https://github.com/david-gpu/srez)           | Tensorflow |
|        [waifu2x](https://github.com/nagadomi/waifu2x)        |   Torch    |
|        [BasicSR](https://github.com/xinntao/BasicSR)         |  PyTorch   |
| [super-resolution](https://github.com/icpm/super-resolution) |  PyTorch   |
| [VideoSuperResolution](https://github.com/LoSealL/VideoSuperResolution) | Tensorflow |
| [video-super-resolution](https://github.com/thangvubk/video-super-resolution) |  Pytorch   |
|[MMSR](https://github.com/open-mmlab/mmsr) | PyTorch |



## Datasets

Note this table is referenced from [here](https://github.com/LoSealL/VideoSuperResolution#link-of-datasets).

|     Name     |   Usage    |                             Link                             |                        Comments                        |
| :----------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------: |
|     Set5     |    Test    | [download](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip) | [jbhuang0604](https://github.com/jbhuang0604/SelfExSR) |
|    SET14     |    Test    | [download](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip) | [jbhuang0604](https://github.com/jbhuang0604/SelfExSR) |
|    BSD100    |    Test    | [download](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip) | [jbhuang0604](https://github.com/jbhuang0604/SelfExSR) |
|   Urban100   |    Test    | [download](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip) | [jbhuang0604](https://github.com/jbhuang0604/SelfExSR) |
|   Manga109   |    Test    |       [website](http://www.manga109.org/ja/index.html)       |                                                        |
|   SunHay80   |    Test    | [download](https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip) | [jbhuang0604](https://github.com/jbhuang0604/SelfExSR) |
|    BSD300    | Train/Val  | [download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz) |                                                        |
|    BSD500    | Train/Val  | [download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) |                                                        |
|   91-Image   |   Train    | [download](http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar) |                          Yang                          |
|  DIV2K2017   | Train/Val  |     [website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)     |                       NTIRE2017                        |
|  Flickr2K   |   Train  |     [download](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)     |     |
|   Real SR    | Train/Val  | [website](https://competitions.codalab.org/competitions/21439#participate) |    NTIRE2019                        |
|   Waterloo   |   Train    |   [website](https://ece.uwaterloo.ca/~k29ma/exploration/)    |                                                        |
|     VID4     |    Test    | [download](https://people.csail.mit.edu/celiu/CVPR2011/videoSR.zip) |                        4 videos                        |
|    MCL-V     |   Train    |        [website](http://mcl.usc.edu/mcl-v-database/)         |                       12 videos                        |
|    GOPRO     | Train/Val  | [website](https://github.com/SeungjunNah/DeepDeblur_release) |                   33 videos, deblur                    |
|    CelebA    |   Train    | [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  |                      Human faces                       |
|    Sintel    | Train/Val  |       [website](http://sintel.is.tue.mpg.de/downloads)       |                      Optical flow                      |
| FlyingChairs |   Train    | [website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs) |                      Optical flow                      |
|  Vimeo-90k   | Train/Test |           [website](http://toflow.csail.mit.edu/)            |                     90k HQ videos                      |
|  SR-RAW   | Train/Test |           [website](https://ceciliavision.github.io/project-pages/project-zoom.html)            |                      raw sensor image dataset                      |
|  W2S   | Train/Test |           [arxiv](https://arxiv.org/pdf/2003.05961.pdf)            |     A Joint Denoising and Super-Resolution Dataset                      |

#### Dataset collections

[Benckmark and DIV2K](https://drive.google.com/drive/folders/1-99XFJs_fvQ2wFdxXrnJFcRRyPJYKN0K): Set5, Set14, B100, Urban100, Manga109, DIV2K2017 include bicubic downsamples with x2,3,4,8

[SR_testing_datasets](https://www.kaggle.com/msahebi/super-resolution#SR_testing_datasets.zip): Test: Set5, Set14, B100, Urban100, Manga109, Historical; Train: T91,General100, BSDS200



## paper

### Non-DL based approach

SCSR: TIP2010, Jianchao Yang et al.[paper](https://ieeexplore.ieee.org/document/5466111/?arnumber=5466111), [code](http://www.ifp.illinois.edu/~jyang29/)

ANR: ICCV2013, Radu Timofte et al. [paper](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-ICCV-2013.pdf), [code](http://www.vision.ee.ethz.ch/~timofter/ICCV2013_ID1774_SUPPLEMENTARY/index.html)

A+: ACCV 2014, Radu Timofte et al. [paper](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-ACCV-2014.pdf), [code](http://www.vision.ee.ethz.ch/~timofter/ACCV2014_ID820_SUPPLEMENTARY/)

IA: CVPR2016, Radu Timofte et al. [paper](http://www.vision.ee.ethz.ch/~timofter/publications/Timofte-CVPR-2016.pdf)

SelfExSR: CVPR2015, Jia-Bin Huang et al. [paper](https://uofi.box.com/shared/static/8llt4ijgc39n3t7ftllx7fpaaqi3yau0.pdf), [code](https://github.com/jbhuang0604/SelfExSR)

NBSRF: ICCV2015, Jordi Salvador et al. [paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Salvador_Naive_Bayes_Super-Resolution_ICCV_2015_paper.pdf)

RFL: ICCV2015, Samuel Schulter et al [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schulter_Fast_and_Accurate_2015_CVPR_paper.pdf), [code](<https://www.tugraz.at/institute/icg/research/team-bischof/samuel-schulter/>)

### DL based approach

Note this table is referenced from [here](https://github.com/LoSealL/VideoSuperResolution/blob/master/README.md#network-list-and-reference-updating)

#### 2014-2016

| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SRCNN                  | [ECCV14](https://arxiv.org/abs/1501.00092)                   | [Keras](https://github.com/qobilidop/srcnn)                  | Kaiming                                                      |
| RAISR                  | [arXiv](https://arxiv.org/abs/1606.01299)                    | -                                                            | Google, Pixel 3                                              |
| ESPCN                  | [CVPR16](https://arxiv.org/abs/1609.05158)                   | [Keras](https://github.com/qobilidop/srcnn)                  | Real time/SISR/**VideoSR**                                   |
| VDSR                   | [CVPR16](https://arxiv.org/abs/1511.04587)                   | [Matlab](http://cv.snu.ac.kr/research/VDSR/)                 | Deep, Residual                                               |
| DRCN                   | [CVPR16](https://arxiv.org/abs/1511.04491)                   | [Matlab](http://cv.snu.ac.kr/research/DRCN/)                 | Recurrent                                                    |

#### 2017
| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DRRN                   | [CVPR17](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) | [Caffe](https://github.com/tyshiwo/DRRN_CVPR17), [PyTorch](https://github.com/jt827859032/DRRN-pytorch) | Recurrent                                                    |
| LapSRN                 | [CVPR17](http://vllab.ucmerced.edu/wlai24/LapSRN/)           | [Matlab](https://github.com/phoenix104104/LapSRN)            | Huber loss                                                   |
| IRCNN                  | [CVPR17](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Learning_Deep_CNN_CVPR_2017_paper.pdf) | [Matlab](https://github.com/cszn/IRCNN)                      |                                                              |
| EDSR                   | [CVPR17](https://arxiv.org/abs/1707.02921)                   | [PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch)       | NTIRE17 Champion                                             |
| BTSRN                  | [CVPR17](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Fan_Balanced_Two-Stage_Residual_CVPR_2017_paper.pdf) | -                                                            | NTIRE17                                                      |
| SelNet                 | [CVPR17](https://ieeexplore.ieee.org/document/8014887)       | -                                                            | NTIRE17                                                      |
| TLSR                   | [CVPR17](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Xu_Fast_and_Accurate_CVPR_2017_paper.pdf) | -                                                            | NTIRE17                                                      |
| SRGAN                  | [CVPR17](https://arxiv.org/abs/1609.04802)                   | [Tensorflow](https://github.com/tensorlayer/srgan)           | 1st proposed GAN                                             |
| VESPCN                 | [CVPR17](https://arxiv.org/abs/1611.05250)                   | -                                                            | **VideoSR**                                                  |
| MemNet                 | [ICCV17](https://arxiv.org/abs/1708.02209)                   | [Caffe](https://github.com/tyshiwo/MemNet)                   |                                                              |
| SRDenseNet             | [ICCV17](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf) | -, [PyTorch](https://github.com/wxywhu/SRDenseNet-pytorch)   | Dense                                                        |
| SPMC                   | [ICCV17](https://arxiv.org/abs/1704.02738)                   | [Tensorflow](https://github.com/jiangsutx/SPMC_VideoSR)      | **VideoSR**                                                  |
| EnhanceNet             | [ICCV17](https://arxiv.org/abs/1612.07919)                   | [TensorFlow](https://github.com/msmsajjadi/EnhanceNet-Code)  | Perceptual Loss                                              |
| PRSR                   | [ICCV17](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dahl_Pixel_Recursive_Super_ICCV_2017_paper.pdf) | [TensorFlow](https://github.com/nilboy/pixel-recursive-super-resolution) | an extension of PixelCNN                                     |
| AffGAN                 | [ICLR17](https://arxiv.org/pdf/1610.04490.pdf)               | -                                                            |                                                              |

#### 2018
| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MS-LapSRN              | [TPAMI18](https://ieeexplore.ieee.org/document/8434354)      | [Matlab](https://github.com/phoenix104104/LapSRN)            | Fast LapSRN                                                  |
| DCSCN                  | [arXiv](https://arxiv.org/abs/1707.05425)                    | [Tensorflow](https://github.com/jiny2001/dcscn-super-resolution) |                                                              |
| IDN                    | [CVPR18](https://arxiv.org/abs/1803.09454)                   | [Caffe](https://github.com/Zheng222/IDN-Caffe)               | Fast                                                         |
| DSRN                   | [CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Han_Image_Super-Resolution_via_CVPR_2018_paper.pdf) | [TensorFlow](https://github.com/WeiHan3/dsrn/tree/db21d57dfab57de3608f0372e749c6488b6b305d) | Dual state，Recurrent                                        |
| RDN                    | [CVPR18](https://arxiv.org/abs/1802.08797)                   | [Torch](https://github.com/yulunzhang/RDN)                   | Deep, BI-BD-DN                                               |
| SRMD                   | [CVPR18](https://arxiv.org/abs/1712.06116)                   | [Matlab](https://github.com/cszn/SRMD)                       | Denoise/Deblur/SR                                            |
| xUnit                  | [CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kligvasser_xUnit_Learning_a_CVPR_2018_paper.pdf) | [PyTorch](https://github.com/kligvasser/xUnit)               | Spatial Activation Function                                  |
| DBPN                   | [CVPR18](https://arxiv.org/abs/1803.02735)                   | [PyTorch](https://github.com/alterzero/DBPN-Pytorch)         | NTIRE18 Champion                                             |
| WDSR                   | [CVPR18](https://arxiv.org/abs/1808.08718)                   | [PyTorch](https://github.com/JiahuiYu/wdsr_ntire2018)，[TensorFlow](https://github.com/ychfan/tf_estimator_barebone/blob/master/docs/super_resolution.md) | NTIRE18 Champion                                             |
| ProSRN                 | [CVPR18](https://arxiv.org/abs/1804.02900)                   | [PyTorch](https://github.com/fperazzi/proSR)                 | NTIRE18                                                      |
| ZSSR                   | [CVPR18](http://www.wisdom.weizmann.ac.il/~vision/zssr/)     | [Tensorflow](https://github.com/assafshocher/ZSSR)           | Zero-shot                                                    |
| FRVSR                  | [CVPR18](https://arxiv.org/abs/1801.04590)                   | [PDF](https://github.com/msmsajjadi/FRVSR)                   | **VideoSR**                                                  |
| DUF                    | [CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf) | [Tensorflow](https://github.com/yhjo09/VSR-DUF)              | **VideoSR**                                                  |
| SFTGAN                 | [CVPR18](https://arxiv.org/abs/1804.02815)                   | [PyTorch](https://github.com/xinntao/SFTGAN)                 |                                                              |
| CARN                   | [ECCV18](https://arxiv.org/abs/1803.08664)                   | [PyTorch](https://github.com/nmhkahn/CARN-pytorch)           | Lightweight                                                  |
| RCAN                   | [ECCV18](https://arxiv.org/abs/1807.02758)                   | [PyTorch](https://github.com/yulunzhang/RCAN)                | Deep, BI-BD-DN                                               |
| MSRN                   | [ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf) | [PyTorch](https://github.com/MIVRC/MSRN-PyTorch)             |                                                              |
| SRFeat                 | [ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf) | [Tensorflow](https://github.com/HyeongseokSon1/SRFeat)       | GAN                                                          |
| TSRN                   | [ECCV18](https://arxiv.org/pdf/1808.00043.pdf)               | [Pytorch](https://github.com/waleedgondal/Texture-based-Super-Resolution-Network) |                                                              |
| ESRGAN                 | [ECCV18](https://arxiv.org/abs/1809.00219)                   | [PyTorch](https://github.com/xinntao/ESRGAN)                 | PRIM18 region 3 Champion                                     |
| EPSR                   | [ECCV18](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Vasu_Analyzing_Perception-Distortion_Tradeoff_using_Enhanced_Perceptual_Super-resolution_Network_ECCVW_2018_paper.pdf) | [PyTorch](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw) | PRIM18 region 1 Champion                                     |
| PESR                   | [ECCV18](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Vu_Perception-Enhanced_Image_Super-Resolution_via_Relativistic_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf) | [PyTorch](https://github.com/thangvubk/PESR)                 | ECCV18 workshop                                              |
| FEQE                   | [ECCV18](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Vu_Fast_and_Efficient_Image_Quality_Enhancement_via_Desubpixel_Convolutional_Neural_ECCVW_2018_paper.pdf) | [Tensorflow](https://github.com/thangvubk/FEQE)              | Fast                                                         |
| NLRN                   | [NIPS18](https://papers.nips.cc/paper/7439-non-local-recurrent-network-for-image-restoration.pdf) | [Tensorflow](https://github.com/Ding-Liu/NLRN)               | Non-local, Recurrent                                         |
| SRCliqueNet            | [NIPS18](https://arxiv.org/abs/1809.04508)                   | -                                                            | Wavelet                                                      |
| CBDNet                 | [arXiv](https://arxiv.org/abs/1807.04686)                    | [Matlab](https://github.com/GuoShi28/CBDNet)                 | Blind-denoise                                                |
| TecoGAN                | [arXiv](http://arxiv.org/abs/1811.09393)                     | [Tensorflow](https://github.com/thunil/TecoGAN)              | **VideoSR** GAN                                              |

#### 2019
| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Adversarial Attacks SR                  | [ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Evaluating_Robustness_of_Deep_Image_Super-Resolution_Against_Adversarial_Attacks_ICCV_2019_paper.pdf) | [PyTorch](https://github.com/IVRL/Kernel-Modeling-Super-Resolution) | 1、对抗攻击对SR的影响。Adversarial Attacks SR                                    |
| KMSR                   | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Kernel_Modeling_Super-Resolution_on_Real_Low-Resolution_Images_ICCV_2019_paper.pdf) | [PyTorch](https://github.com/IVRL/Kernel-Modeling-Super-Resolution) | 2、真实SR，用GAN进行模糊核估计。GAN for blur-kernel estimation                               |
| SROBB                  | [ICCV2019](https://arxiv.org/pdf/1908.07222.pdf)               |                                                              | 3、基于目标的感知损失。Targeted Perceptual Loss                                     |
| WSTSR                 | [ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Deng_Wavelet_Domain_Style_Transfer_for_an_Effective_Perception-Distortion_Tradeoff_in_ICCV_2019_paper.pdf)               |                                                              | 4、小波域风格迁移SR，Wavelet Domain Style Transfer                                     |
| RealSR                   | [ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cai_Toward_Real-World_Single_Image_Super-Resolution_A_New_Benchmark_and_a_ICCV_2019_paper.pdf)                   | [PyTorch](https://github.com/csjcai/RealSR)         | 5、真实场景的SR                                                 |
| RankSRGAN              | [ICCV2019](https://arxiv.org/pdf/1908.06382.pdf)               | [PyTorch](https://github.com/WenlongZhang0724/RankSRGAN)     | 6、oral, rank-content loss                                      |
| PFNL                   | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yi_Progressive_Fusion_Video_Super-Resolution_Network_via_Exploiting_Non-Local_Spatio-Temporal_Correlations_ICCV_2019_paper.pdf) | [Tensorflow](https://github.com/psychopa4/PFNL)              | 7、视频SR, oral,Non-Local Spatio-Temporal Correlations      |
| Deep SR-ITM            | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kim_Deep_SR-ITM_Joint_Learning_of_Super-Resolution_and_Inverse_Tone-Mapping_for_ICCV_2019_paper.pdf) | [matlab](https://github.com/sooyekim/Deep-SR-ITM)            | 8、SDR to HDR, 4K SR                                            |
| EBRN                   | [ICCV2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qiu_Embedded_Block_Residual_Network_A_Recursive_Restoration_Model_for_Single-Image_ICCV_2019_paper.pdf) |                                                              | 9、嵌入残差，刷PSNR指标。Embedded Block Residual Network                              |
| MSG              | [ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Voynov_Perceptual_Deep_Depth_Super-Resolution_ICCV_2019_paper.pdf)               | [project](http://adase.group/3ddl/projects/perceptual-depth-sr/)     | 10、3D,立体SR                                      |
| TwoStreamSR              | [ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Two-Stream_Action_Recognition-Oriented_Video_Super-Resolution_ICCV_2019_paper.pdf)               | [PyTorch](https://github.com/AlanZhang1995/TwoStreamSR)     | 11、动作识别视频SR。oral, rank-content loss                                      |
| PixTransform              | [ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/de_Lutio_Guided_Super-Resolution_As_Pixel-to-Pixel_Transformation_ICCV_2019_paper.pdf)               | [PyTorch](https://github.com/riccardodelutio/PixTransform)     | 12、Guided Super-Resolution as Pixel-to-Pixel Transformation                                     |
| RBPN                   | [CVPR19](https://arxiv.org/abs/1903.10128)                   | [PyTorch](https://github.com/alterzero/RBPN-PyTorch)         | **VideoSR**                                                  |
| SRFBN                  | [CVPR19](https://arxiv.org/abs/1903.09814)                   | [PyTorch](https://github.com/Paper99/SRFBN_CVPR19)           | Feedback                                                     |
| AdaFM                  | [CVPR19](https://arxiv.org/pdf/1904.08118.pdf)               | [PyTorch](https://github.com/hejingwenhejingwen/AdaFM)       | Adaptive Feature Modification Layers                         |
| MoreMNAS               | [arXiv](https://arxiv.org/pdf/1901.01074.pdf)                | -                                                            | Lightweight，NAS                                             |
| FALSR                  | [arXiv](https://arxiv.org/pdf/1901.07261.pdf)                | [TensorFlow](https://ieeexplore.ieee.org/document/8434354)   | Lightweight，NAS                                             |
| Meta-SR                | [CVPR19](https://arxiv.org/pdf/1903.00875.pdf)               | [PyTorch](https://github.com/XuecaiHu/Meta-SR-Pytorch)       | Arbitrary Magnification                                      |
| AWSRN                  | [arXiv](https://arxiv.org/abs/1904.02358)                    | [PyTorch](https://github.com/ChaofWang/AWSRN)                | Lightweight                                                  |
| OISR                   | [CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_ODE-Inspired_Network_Design_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf) | [PyTorch](https://github.com/HolmesShuan/OISR-PyTorch)       | ODE-inspired Network                                         |
| DPSR                   | [CVPR19](https://arxiv.org/pdf/1903.12529.pdf)               | [PyTorch](https://github.com/cszn/DPSR)                      |                                                              |
| DNI                    | [CVPR19](https://arxiv.org/pdf/1811.10515.pdf)               | [PyTorch](https://github.com/xinntao/DNI)                    |                                                              |
| MAANet                 | [arXiv](https://arxiv.org/abs/1904.06252)                    |                                                              | Multi-view Aware Attention                                   |
| RNAN                   | [ICLR19](https://openreview.net/pdf?id=HkeGhoA5FX)           | [PyTorch](https://github.com/yulunzhang/RNAN)                | Residual Non-local Attention                                 |
| FSTRN                  | [CVPR19](https://arxiv.org/pdf/1904.02870.pdf)               | -                                                            | **VideoSR**, fast spatio-temporal residual block             |
| MsDNN                  | [arXiv](https://arxiv.org/pdf/1904.10698.pdf)                | [TensorFlow](https://github.com/shangqigao/gsq-image-SR)     | NTIRE19  real SR  21th place                                 |
| SAN                    | [CVPR19](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-SAN.pdf) | [Pytorch](https://github.com/daitao/SAN)                     | Second-order Attention,cvpr19 oral                           |
| EDVR                   | [CVPRW19](https://arxiv.org/pdf/1905.02716.pdf)              | [Pytorch](https://github.com/xinntao/EDVR)                   | **Video**, NTIRE19 video restoration and enhancement champions |
| Ensemble for VSR       | [CVPRW19](https://arxiv.org/pdf/1905.02462.pdf)              | -                                                            | **VideoSR**, NTIRE19 video SR 2nd place                      |
| TENet                  | [arXiv](https://arxiv.org/pdf/1905.02538.pdf)                | [Pytorch](https://github.com/guochengqian/TENet)             | a Joint Solution for Demosaicking, Denoising and Super-Resolution |
| MCAN                   | [arXiv](https://arxiv.org/pdf/1903.07949.pdf)                | [Pytorch](https://github.com/macn3388/MCAN)                  | Matrix-in-matrix CAN, Lightweight                            |
| IKC&SFTMD              | [CVPR19](https://arxiv.org/pdf/1904.03377.pdf)               | -                                                            | Blind Super-Resolution                                       |
| SRNTT                  | [CVPR19](https://arxiv.org/pdf/1903.00834.pdf)               | [TensorFlow](https://github.com/ZZUTK/SRNTT)                 | Neural Texture Transfer                                      |
| RawSR                  | [CVPR19](https://arxiv.org/pdf/1905.12156.pdf)               | [TensorFlow](https://drive.google.com/file/d/1yvCceNAgt4UsxZXahPFBkuL1JXyfgr8B/view) | Real Scene Super-Resolution, Raw Images                      |
| resLF                  | [CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Residual_Networks_for_Light_Field_Image_Super-Resolution_CVPR_2019_paper.pdf) |                                                              | Light field                                                  |
| CameraSR               | [CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Camera_Lens_Super-Resolution_CVPR_2019_paper.pdf) |                                                              | realistic image SR                                           |
| ORDSR                  | [TIP](https://arxiv.org/pdf/1904.10082.pdf)                  | [model](https://github.com/tT0NG/ORDSR)                      | DCT domain SR                                                |
| U-Net                  | [CVPRW19](https://arxiv.org/pdf/1906.04809.pdf)              |                                                              | NTIRE19  real SR  2nd place, U-Net,MixUp,Synthesis           |
| DRLN                   | [arxiv](https://arxiv.org/pdf/1906.12021.pdf)                |                                                              | Densely Residual Laplacian Super-Resolution                  |
| EDRN                   | [CVPRW19](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Cheng_Encoder-Decoder_Residual_Network_for_Real_Super-Resolution_CVPRW_2019_paper.pdf) | [Pytorch](https://github.com/yyknight/NTIRE2019_EDRN)        | NTIRE19  real SR  9th places                                 |
| FC2N                   | [arXiv](https://arxiv.org/pdf/1907.03221.pdf)                |                                                              | Fully Channel-Concatenated                                   |
| GMFN                   | [BMVC2019](https://arxiv.org/pdf/1907.04253.pdf)             | [Pytorch](https://github.com/liqilei/GMFN)                   | Gated Multiple Feedback                                      |
| CNN&TV-TV Minimization | [BMVC2019](https://arxiv.org/pdf/1907.05380.pdf)             |                                                              | TV-TV Minimization                                           |
| HRAN                   | [arXiv](https://arxiv.org/pdf/1907.05514.pdf)                |                                                              | Hybrid Residual Attention Network                            |
| PPON                   | [arXiv](https://arxiv.org/pdf/1907.10399.pdf)                | [code](https://github.com/Zheng222/PPON)                     | Progressive Perception-Oriented Network                      |
| edge-informed          | [ICCVW19](https://arxiv.org/pdf/1909.05305.pdf)              | [PyTorch](https://github.com/knazeri/edge-informed-sisr)     | Edge-Informed Single Image Super-Resolution                  |
| s-LWSR                 | [arxiv](https://arxiv.org/pdf/1909.10774.pdf)                |                                                              | Lightweight                                                  |
| DNLN                   | [arxiv](https://arxiv.org/pdf/1909.10692.pdf)                |                                                              | **Video SR** Deformable Non-local Network                    |
| MGAN                   | [arxiv](https://arxiv.org/pdf/1909.11937.pdf)                |                                                              | Multi-grained Attention Networks                             |
| IMDN                   | [ACM MM 2019](https://arxiv.org/pdf/1909.11856.pdf)          | [PyTorch](https://github.com/Zheng222/IMDN)                  | AIM19  Champion                                              |
| ESRN                   | [arxiv](https://arxiv.org/pdf/1909.11409.pdf)                |                                                              | NAS                                                          |
| feature SR             | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Noh_Better_to_Follow_Follow_to_Be_Better_Towards_Precise_Supervision_ICCV_2019_paper.pdf) |                                                              | Super-Resolution for Small Object Detection                  |
| STFAN                  | [ICCV19](https://arxiv.org/pdf/1904.12257.pdf)               | [PyTorch](https://github.com/sczhou/STFAN)                   | **Video Deblurring**                                         |
| CFSNet                 | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_CFSNet_Toward_a_Controllable_Feature_Space_for_Image_Restoration_ICCV_2019_paper.pdf) | [PyTorch](https://github.com/qibao77/CFSNet)                 | Controllable Feature                                         |
| FSRnet                 | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gu_Fast_Image_Restoration_With_Multi-Bin_Trainable_Linear_Units_ICCV_2019_paper.pdf) |                                                              | Multi-bin Trainable Linear Units                             |
| SAM+VAM                | [ICCVW19](https://arxiv.org/pdf/1911.08711.pdf)              |                                                              |                                                              |
| SinGAN                 | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf) |    [PyTorch](https://github.com/tamarott/SinGAN)              | bestpaper,  train from single image   |
|KernelGAN        |[NIPS 2019(oral)](http://papers.nips.cc/paper/8321-blind-super-resolution-kernel-estimation-using-an-internal-gan.pdf)   |[PyTorch](http://www.wisdom.weizmann.ac.il/~vision/kernelgan/)   | 无监督GAN,值得学习       |

#### 2020
| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DRN        | [CVPR 2020](https://arxiv.org/pdf/2003.07018.pdf)            | [PyTorch](https://github.com/guoyongcs/DRN)                                                             | 1、对偶回归，面向真实LR图像Dual Regression, SISR STOA                     |
| EventSR        | [CVPR 2020](https://arxiv.org/pdf/2003.07640.pdf)            |                                                             | 2、针对event camera数据进行SR。split three phases                     |
| Pseudo-Supervision SR  | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Maeda_Unpaired_Image_Super-Resolution_Using_Pseudo-Supervision_CVPR_2020_paper.pdf)    | | 3、伪监督SR |
| Correction filter     | [CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Abu_Hussein_Correction_Filter_for_Single_Image_Super-Resolution_Robustifying_Off-the-Shelf_Deep_Super-Resolvers_CVPR_2020_paper.pdf)            |    | 4、在现有的模型基础上进行矫正核，盲SR |
| RFANet               | [CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Residual_Feature_Aggregation_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf)            |    | 5、残差特征聚合，非盲SR |
| `USRNet`       | [CVPR 2020](https://arxiv.org/pdf/2003.10428.pdf)            | [PyTorch](https://github.com/cszn/USRNet)                                                             | 6、深度展开网络SR                   |
| CSNLN               | [CVPR 2020](https://arxiv.org/pdf/2006.01424.pdf)            | [PyTorch](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention)         | 7、跨尺度非局部注意力，非盲SR,bicubic下采样 Similar to PANet  |
| `UDVD`      | [CVPR 2020](https://arxiv.org/pdf/2004.06965.pdf)            |   | 8、采用统一的模型来应对不同的退化场景          |
| `TTSR`               | [CVPR 2020](https://arxiv.org/pdf/2006.04139.pdf)            | [PyTorch](https://github.com/FuzhiYang/TTSR)         | 9、纹理迁移Texture Transformer |
| SSEN     | [CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shim_Robust_Reference-Based_Super-Resolution_With_Similarity-Aware_Deformable_Convolution_CVPR_2020_paper.pdf)    |   | 10、Similarity-Aware Deformable Convolution |
| TDAN                   | [CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tian_TDAN_Temporally-Deformable_Alignment_Network_for_Video_Super-Resolution_CVPR_2020_paper.pdf)                |  [PyTorch](https://github.com/YapengTian/TDAN-VSR-CVPR-2020)                                                      | 11、**视频SR**，Deformable Align                                |
| Zooming Slow-Mo        | [CVPR 2020](https://arxiv.org/pdf/2002.11616.pdf)            | [PyTorch](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)                                                             | 12、视频SR，joint VFI and SR，one-stage，  deformable ConvLSTM                     |
| VSR_TGA     | [CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Isobe_Video_Super-Resolution_With_Temporal_Group_Attention_CVPR_2020_paper.pdf)    | [code](https://github.com/junpan19/VSR_TGA)   | 13、视频SR，Temporal Group Attention, Fast Spatial Alignment |
|LESR        |[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Meishvili_Learning_to_Have_an_Ear_for_Face_Super-Resolution_CVPR_2020_paper.pdf)   |[PyTorch](https://gmeishvili.github.io/ear_for_face_super_resolution/index.html)   | 14、人脸SR，通过一只耳朵的声音来帮助人脸SR      |
|DICGAN        |[CVPR 2020](https://arxiv.org/pdf/2003.13063.pdf)   |[PyTorch](https://github.com/Maclory/Deep-Iterative-Collaboration)   | 15、人脸SR        |
|AIR        |[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Song_Channel_Attention_Based_Iterative_Residual_Learning_for_Depth_Map_Super-Resolution_CVPR_2020_paper.pdf)   | | 16、深度图超分辨率     |
|LFSR        |[CVPR 2020](https://arxiv.org/pdf/2004.12811.pdf)   |[PyTorch](https://github.com/jingjin25/LFSSR-ATO)  | 17、光场图像SR        |
|UAL        |[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Unsupervised_Adaptation_Learning_for_Hyperspectral_Imagery_Super-Resolution_CVPR_2020_paper.pdf)   |  [PyTorch](https://github.com/JiangtaoNie/UAL)| 18、高光谱图像SR     |
| MZSR        | [CVPR 2020](https://arxiv.org/pdf/2002.12213.pdf)            |      [PyTorch](https://github.com/JWSoh/MZSR)                                                   
