# Awesome-Super-Resolution
Quick navigation
repositories
Awesome paper list
Awesome repos
Datasets
Dataset collections
paper
Non-DL based approach
DL based approach
2014-2016
2017
2018
2019
2020
Super Resolution workshop papers
NTIRE17
NTIRE18
PIRM18
NTIRE19
AIM19
Super Resolution survey
Awesome-Super-Resolution（in progress）
Collect some super-resolution related papers, data and repositories.

repositories
Awesome paper list:
Single-Image-Super-Resolution

Super-Resolution.Benckmark

Video-Super-Resolution

VideoSuperResolution

Awesome Super-Resolution

Awesome-LF-Image-SR

Awesome-Stereo-Image-SR

AI-video-enhance

Awesome repos:
repo	Framework
EDSR-PyTorch	PyTorch
Image-Super-Resolution	Keras
image-super-resolution	Keras
Super-Resolution-Zoo	MxNet
super-resolution	Keras
neural-enhance	Theano
srez	Tensorflow
waifu2x	Torch
BasicSR	PyTorch
super-resolution	PyTorch
VideoSuperResolution	Tensorflow
video-super-resolution	Pytorch
MMSR	PyTorch
Datasets
Note this table is referenced from here.

Name	Usage	Link	Comments
Set5	Test	download	jbhuang0604
SET14	Test	download	jbhuang0604
BSD100	Test	download	jbhuang0604
Urban100	Test	download	jbhuang0604
Manga109	Test	website	
SunHay80	Test	download	jbhuang0604
BSD300	Train/Val	download	
BSD500	Train/Val	download	
91-Image	Train	download	Yang
DIV2K2017	Train/Val	website	NTIRE2017
Flickr2K	Train	download	
Real SR	Train/Val	website	NTIRE2019
Waterloo	Train	website	
VID4	Test	download	4 videos
MCL-V	Train	website	12 videos
GOPRO	Train/Val	website	33 videos, deblur
CelebA	Train	website	Human faces
Sintel	Train/Val	website	Optical flow
FlyingChairs	Train	website	Optical flow
Vimeo-90k	Train/Test	website	90k HQ videos
SR-RAW	Train/Test	website	raw sensor image dataset
W2S	Train/Test	arxiv	A Joint Denoising and Super-Resolution Dataset
Dataset collections
Benckmark and DIV2K: Set5, Set14, B100, Urban100, Manga109, DIV2K2017 include bicubic downsamples with x2,3,4,8

SR_testing_datasets: Test: Set5, Set14, B100, Urban100, Manga109, Historical; Train: T91,General100, BSDS200

paper
Non-DL based approach
SCSR: TIP2010, Jianchao Yang et al.paper, code

ANR: ICCV2013, Radu Timofte et al. paper, code

A+: ACCV 2014, Radu Timofte et al. paper, code

IA: CVPR2016, Radu Timofte et al. paper

SelfExSR: CVPR2015, Jia-Bin Huang et al. paper, code

NBSRF: ICCV2015, Jordi Salvador et al. paper

RFL: ICCV2015, Samuel Schulter et al paper, code

DL based approach
Note this table is referenced from here

2014-2016
Model	Published	Code	Keywords
SRCNN	ECCV14	Keras	Kaiming
RAISR	arXiv	-	Google, Pixel 3
ESPCN	CVPR16	Keras	Real time/SISR/VideoSR
VDSR	CVPR16	Matlab	Deep, Residual
DRCN	CVPR16	Matlab	Recurrent
2017
Model	Published	Code	Keywords
DRRN	CVPR17	Caffe, PyTorch	Recurrent
LapSRN	CVPR17	Matlab	Huber loss
IRCNN	CVPR17	Matlab	
EDSR	CVPR17	PyTorch	NTIRE17 Champion
BTSRN	CVPR17	-	NTIRE17
SelNet	CVPR17	-	NTIRE17
TLSR	CVPR17	-	NTIRE17
SRGAN	CVPR17	Tensorflow	1st proposed GAN
VESPCN	CVPR17	-	VideoSR
MemNet	ICCV17	Caffe	
SRDenseNet	ICCV17	-, PyTorch	Dense
SPMC	ICCV17	Tensorflow	VideoSR
EnhanceNet	ICCV17	TensorFlow	Perceptual Loss
PRSR	ICCV17	TensorFlow	an extension of PixelCNN
AffGAN	ICLR17	-	
2018
Model	Published	Code	Keywords
MS-LapSRN	TPAMI18	Matlab	Fast LapSRN
DCSCN	arXiv	Tensorflow	
IDN	CVPR18	Caffe	Fast
DSRN	CVPR18	TensorFlow	Dual state，Recurrent
RDN	CVPR18	Torch	Deep, BI-BD-DN
SRMD	CVPR18	Matlab	Denoise/Deblur/SR
xUnit	CVPR18	PyTorch	Spatial Activation Function
DBPN	CVPR18	PyTorch	NTIRE18 Champion
WDSR	CVPR18	PyTorch，TensorFlow	NTIRE18 Champion
ProSRN	CVPR18	PyTorch	NTIRE18
ZSSR	CVPR18	Tensorflow	Zero-shot
FRVSR	CVPR18	PDF	VideoSR
DUF	CVPR18	Tensorflow	VideoSR
TDAN	arXiv	-	VideoSR，Deformable Align
SFTGAN	CVPR18	PyTorch	
CARN	ECCV18	PyTorch	Lightweight
RCAN	ECCV18	PyTorch	Deep, BI-BD-DN
MSRN	ECCV18	PyTorch	
SRFeat	ECCV18	Tensorflow	GAN
TSRN	ECCV18	Pytorch	
ESRGAN	ECCV18	PyTorch	PRIM18 region 3 Champion
EPSR	ECCV18	PyTorch	PRIM18 region 1 Champion
PESR	ECCV18	PyTorch	ECCV18 workshop
FEQE	ECCV18	Tensorflow	Fast
NLRN	NIPS18	Tensorflow	Non-local, Recurrent
SRCliqueNet	NIPS18	-	Wavelet
CBDNet	arXiv	Matlab	Blind-denoise
TecoGAN	arXiv	Tensorflow	VideoSR GAN
2019
Model	Published	Code	Keywords
RBPN	CVPR19	PyTorch	VideoSR
SRFBN	CVPR19	PyTorch	Feedback
AdaFM	CVPR19	PyTorch	Adaptive Feature Modification Layers
MoreMNAS	arXiv	-	Lightweight，NAS
FALSR	arXiv	TensorFlow	Lightweight，NAS
Meta-SR	CVPR19	PyTorch	Arbitrary Magnification
AWSRN	arXiv	PyTorch	Lightweight
OISR	CVPR19	PyTorch	ODE-inspired Network
DPSR	CVPR19	PyTorch	
DNI	CVPR19	PyTorch	
MAANet	arXiv		Multi-view Aware Attention
RNAN	ICLR19	PyTorch	Residual Non-local Attention
FSTRN	CVPR19	-	VideoSR, fast spatio-temporal residual block
MsDNN	arXiv	TensorFlow	NTIRE19 real SR 21th place
SAN	CVPR19	Pytorch	Second-order Attention,cvpr19 oral
EDVR	CVPRW19	Pytorch	Video, NTIRE19 video restoration and enhancement champions
Ensemble for VSR	CVPRW19	-	VideoSR, NTIRE19 video SR 2nd place
TENet	arXiv	Pytorch	a Joint Solution for Demosaicking, Denoising and Super-Resolution
MCAN	arXiv	Pytorch	Matrix-in-matrix CAN, Lightweight
IKC&SFTMD	CVPR19	-	Blind Super-Resolution
SRNTT	CVPR19	TensorFlow	Neural Texture Transfer
RawSR	CVPR19	TensorFlow	Real Scene Super-Resolution, Raw Images
resLF	CVPR19		Light field
CameraSR	CVPR19		realistic image SR
ORDSR	TIP	model	DCT domain SR
U-Net	CVPRW19		NTIRE19 real SR 2nd place, U-Net,MixUp,Synthesis
DRLN	arxiv		Densely Residual Laplacian Super-Resolution
EDRN	CVPRW19	Pytorch	NTIRE19 real SR 9th places
FC2N	arXiv		Fully Channel-Concatenated
GMFN	BMVC2019	Pytorch	Gated Multiple Feedback
CNN&TV-TV Minimization	BMVC2019		TV-TV Minimization
HRAN	arXiv		Hybrid Residual Attention Network
PPON	arXiv	code	Progressive Perception-Oriented Network
SROBB	ICCV19		Targeted Perceptual Loss
RankSRGAN	ICCV19	PyTorch	oral, rank-content loss
edge-informed	ICCVW19	PyTorch	Edge-Informed Single Image Super-Resolution
s-LWSR	arxiv		Lightweight
DNLN	arxiv		Video SR Deformable Non-local Network
MGAN	arxiv		Multi-grained Attention Networks
IMDN	ACM MM 2019	PyTorch	AIM19 Champion
ESRN	arxiv		NAS
PFNL	ICCV19	Tensorflow	VideoSR oral,Non-Local Spatio-Temporal Correlations
EBRN	ICCV19		Embedded Block Residual Network
Deep SR-ITM	ICCV19	matlab	SDR to HDR, 4K SR
feature SR	ICCV19		Super-Resolution for Small Object Detection
STFAN	ICCV19	PyTorch	Video Deblurring
KMSR	ICCV19	PyTorch	GAN for blur-kernel estimation
CFSNet	ICCV19	PyTorch	Controllable Feature
FSRnet	ICCV19		Multi-bin Trainable Linear Units
SAM+VAM	ICCVW19		
SinGAN	ICCV19	PyTorch	bestpaper, train from single image
2020
Model	Published	Code	Keywords
FISR	AAAI 2020	TensorFlow	Video joint VFI-SR method,Multi-scale Temporal Loss
ADCSR	arxiv		
SCN	AAAI 2020		Scale-wise Convolution
LSRGAN	arxiv		Latent Space Regularization for srgan
Zooming Slow-Mo	CVPR 2020	PyTorch	joint VFI and SR，one-stage， deformable ConvLSTM
MZSR	CVPR 2020		Meta-Transfer Learning, Zero-Shot
VESR-Net	arxiv		Youku Video Enhancement and Super-Resolution Challenge Champion
blindvsr	arxiv	PyTorch	Motion blur estimation
HNAS-SR	arxiv	PyTorch	Hierarchical Neural Architecture Search, Lightweight
DRN	CVPR 2020	PyTorch	Dual Regression, SISR STOA
SFM	arxiv	PyTorch	Stochastic Frequency Masking, Improve method
EventSR	CVPR 2020		split three phases
USRNet	CVPR 2020	PyTorch	
PULSE	CVPR 2020		Self-Supervised
SPSR	CVPR 2020	Code	Gradient Guidance, GAN
DASR	arxiv	Code	Real-World Image Super-Resolution, Unsupervised SuperResolution, Domain Adaptation.
STVUN	arxiv	PyTorch	Video Super-Resolution, Video Frame Interpolation, Joint space-time upsampling
AdaDSR	arxiv	PyTorch	Adaptive Inference
Scale-Arbitrary SR	arxiv	Code	Scale-Arbitrary Super-Resolution, Knowledge Transfer
DeepSEE	arxiv	Code	Extreme super-resolution,32× magnification
CutBlur	CVPR 2020	PyTorch	SR Data Augmentation
UDVD	CVPR 2020		Unified Dynamic Convolutional，SISR and denoise
DIN	IJCAI-PRICAI 2020		SISR，asymmetric co-attention
PANet	arxiv	PyTorch	Pyramid Attention
SRResCGAN	arxiv	PyTorch	
Super Resolution workshop papers
NTIRE17 papers
NTIRE18 papers
PIRM18 Web
NTIRE19 papers
AIM19 papers
NTIRE20 NTIRE 2020
---Image and Video Deblurring

---Perceptual Extreme Super-Resolution

---Real-World Image Super-Resolution

Super Resolution survey
[1] Wenming Yang, Xuechen Zhang, Yapeng Tian, Wei Wang, Jing-Hao Xue. Deep Learning for Single Image Super-Resolution: A Brief Review. arxiv, 2018. paper

[2]Saeed Anwar, Salman Khan, Nick Barnes. A Deep Journey into Super-resolution: A survey. arxiv, 2019.paper

[3]Wang, Z., Chen, J., & Hoi, S. C. (2019). Deep learning for image super-resolution: A survey. arXiv preprint arXiv:1902.06068.paper
