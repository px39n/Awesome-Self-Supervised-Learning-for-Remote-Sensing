


# Awesome-Self-Supervised-Learning-for-Remote-Sensing [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)
List of reference,algorithms, applications in SSL in RS (**contribution are welcome**)
![](https://i.imgur.com/kBwKJH3.png)

# Table of Contents
- [Background](#background)
  * [Literature Review](#literature-review)
  * [Basic Concept](#basic-concept)
- [Current Challengings](#current-challengings)
- [Algorithm](#algorithm)
  * [Pretext](#pretext)
    + [Generative](#generative)
    + [Contrastive](#contrastive)
    + [GANs](#gans)
  * [Backbone Network Architecture](#backbone-network-architecture)
  * [Technology](#technology)
  * [Mentioned Frameworks](#mentioned-frameworks)
    + [Traditional CV](#traditional-cv)
    + [Remote senisng](#remote-senisng)
- [Dataset](#dataset)
  * [Fine Tune Dataset](#fine-tune-dataset)
  * [Earth Observation Datasets](#earth-observation-datasets)
- [Quality](#quality)
- [Terms](#terms)


# Background
 SSL in Computer Vision (CV):
- Supervised learning requires **expensive manual labeling** and is susceptible to errors and attacks.
- Deep neural networks rely heavily on the size and **quality** of training data, limiting their applicability in real-world scenarios.
- SSL is a promising alternative that is data efficient and enhances generalization capabilities.
 
SSL in Remote Sensing (RS):
- Machine/deep learning requires a **large amount** of training data, which is available in open-access Earth observation.
- Annotating such data is **time-consuming** and frequently updated, making it difficult to generate perfect labels.
- Pre-training models in transfer learning have become popular for downstream tasks with **insufficient labels**.
- SSL bridges the gap between a lack of quality labels and increasing amounts of remote sensing data by utilizing unlabeled data to learn valuable information.

## Literature Review

**Trend in SSL** 

| Year | Title                                                                       | Cite | Link                                                                       |
| ---- | --------------------------------------------------------------------------- | ---- | -------------------------------------------------------------------------- |
| 2020 | Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey | 510  | [[1](https://pubmed.ncbi.nlm.nih.gov/32386141/)]                           |
| 2020 | Self-Supervised Learning: Generative or Contrastive                         | 577  | [[3](https://ieeexplore.ieee.org/document/9128962)]                        |
| 2021 | Review on self-supervised image recognition using deep neural networks      | 46   | [[2](https://www.sciencedirect.com/science/article/pii/S0950705121003531)] |


**Trend in SSL-Remote Senisng Based** 

| Year | Title                                                                                                          | Cited | Link                                            |
| ---- | -------------------------------------------------------------------------------------------------------------- | ----- | ----------------------------------------------- |
| 2022 | Self-Supervised Learning in Remote Sensing: A review                                                           | 28    | [1](https://doi.org/10.48550/arXiv.2211.08129 ) |
| 2022 | Self-supervised remote sensing feature learning                                                                | 1     | [2]( https://arxiv.org/abs/2211.07467)          |
| 2022 | Self-Supervised Learning for Scene Classification in Remote Sensing: Current State of the Art and Perspectives | 0      |[3](https://mdpi.com/2072-4292/14/16/3995)                                                |
## Basic Concept

A typical self-supervised learning (SSL) framework is based on: 

**specific pretext task**, which is designed to learn useful representations of input data without explicit labels. The pretext task defines the training objective of the SSL framework and can take various forms, such as predicting the relative position of two patches from the same image or predicting the rotation angle of an image.

**Architecture**: the SSL framework can use any architecture to implement the model. Like Residual, transformer etc,.

**SSL Technology**: To optimize the model, SSL frameworks use a variety of technologies and loss functions, including negative sampling, memory banks, contrastive learning, and augmentation strategies. These technologies are designed to reduce overfitting, improve the quality of learned representations, and make the learning process more efficient.

With all the background, we can **define** SSL is :

1.  Utilize advanced SSL [[#Technology]] to train specific [[#Backbone Network Architecture]] by fitting them to the given [[#Pretext Type]].
2.  Utilize the trained architectures as pre-trained models for **downstream tasks**.



# Current Challengings
Challenges of SSL in Computer Vision:

| Challenge                                         | Description                                                                                                                                                                      |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Early degradation of contrastive learning methods | While methods like MoCo and SimCLR are approaching supervised learning performance in computer vision, their performance is typically limited to classification problems.        |
| Transfer to downstream tasks                      | There is an inherent gap between pre-training and downstream tasks, and the process of choosing proxy tasks for pre-training may be heuristic and tricky.                        |
| Model collapse                                    | Model collapse is a major challenge in SSL, especially for modern contrastive learning methods.                                                                                  |
| Proxy tasks and data augmentation                 | Proxy tasks and data augmentation play a crucial role in SSL, but more research is needed to better understand which ones are useful for different types of remote sensing data. |

Challenges of SSL in Remote Sensing:

| Challenge                                 | Description                                                                                                                                                                        |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model collapse                            | Model collapse is a major challenge in SSL, and more research is needed to understand the underlying theory of self-supervised representation learning.                            |
| Proxy tasks and data augmentation         | More research is needed to explore which proxy tasks and data augmentation techniques are useful for different types of remote sensing data.                                       |
| Pre-training datasets                     | Most existing remote sensing datasets are used for supervised learning, and there is a need to explore the use of large-scale unlabeled datasets for self-supervised pre-training. |
| Multi-modal/time self-supervised learning | More research is needed to balance different modalities or timestamps to allow models to learn good representations.                                                               |
| Efficient computation of SSL              | More research is needed to explore efficient data compression, loading, model design, and hardware acceleration to reduce the computational cost of SSL.                           |
| Network backbone                       | While most existing SSL methods use ResNet as their backbone, ViT has shown promising results in SSL and is worth exploring for remote sensing images. |
| Task-oriented weakly supervised learning | SSL not only provides pre-trained models for downstream tasks but also has the potential to bring representation learning online for weakly supervised learning. |

# Algorithm

A typical self-supervised learning (SSL) framework is based on: 

**specific pretext task**, which is designed to learn useful representations of input data without explicit labels. The pretext task defines the training objective of the SSL framework and can take various forms, such as predicting the relative position of two patches from the same image or predicting the rotation angle of an image.

**Architecture**: the SSL framework can use any architecture to implement the model. Like Residual, transformer etc,.

**SSL Technology**: To optimize the model, SSL frameworks use a variety of technologies and loss functions, including negative sampling, memory banks, contrastive learning, and augmentation strategies. These technologies are designed to reduce overfitting, improve the quality of learned representations, and make the learning process more efficient.

## Pretext  
A typical self-supervised learning (SSL) framework is based on a specific pretext task, which is designed to learn useful representations of input data without explicit labels. The pretext task defines the training objective of the SSL framework and can take various forms, such as predicting the relative position of two patches from the same image or predicting the rotation angle of an image.

### Generative

| Category | Pretext Task Type     | Explanation                                                                                                | Model Examples                              |
| -------- | --------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| AE       | Next pixel prediction | Predict the next pixel in an image based on context information.                                           | Denoising Autoencoder, Denoising VAE        |
|          | Image reconstruction  | Reconstruct the whole image from parts or low-quality versions of the image.                               | NICE, RealNVP, Glow, VQ-VAE 2               |
|          | Variational Inference | Encodes the input to a normal distribution, enabling generation of new samples with the same distribution. | Variational Autoencoder (VAE), IWAE, VQ-VAE |

### Contrastive

| Category | Pretext Task Type                  | Explanation                                                                                                                                                                                                             | Model Examples                                                                                                                                                                                        |
| -------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Spatial  | Relative position prediction       | Predict the relative position of different parts of an image based on their spatial relationships.                                                                                                                      | RelativePosition, DeepInfomax, CPC, Rotation-based Relative Position, Context-Aware Correspondence Network, Paired Similarity Comparison (PSC)                                                        |
|          | Jigsaw + Inpainting + Colorization | Solve jigsaw puzzles, inpaint missing parts of an image, and colorize grayscale images by learning their context relationships.                                                                                         | CDJP, DeepFillv2, Context Encoders, Colorization, PatchGan, Semantic Inpainting, Complementary Spatial Feature Imagination (CSFI), JigCA                                                              |
|          | Instance discrimination            | Distinguish between different instances of the same object class by learning their unique features.                                                                                                                     | InstDisc, CMC, MoCo, SimCLR, BYOL, SimSiam, Deep InfoMax, Contrastive Multiview Coding (CMC), Bootstrap Your Own Latent (BYOL), Local Aggregation for Unsupervised Learning (LAUL)                    |
| Spectral | Colorization                       | Colorize grayscale images by learning the relationship between color channels and luminance information.                                                                                                                | Colorization, Linearly Scalable Colorization, Multimodal Unsupervised Image-to-Image Translation (MUNIT), Semantic Colorization, Multi-Scale Style Attentional Generative Adversarial Network (MSGAN) |
| Temporal | Frame order prediction             | Predict the order of frame sequences or video frames based on their temporal relationships.                                                                                                                             | Temporal Order Prediction (TOP), SlowFast, Sequence Memory, Time-contrastive Network (TCN), Learning Correspondence from the Cycle-consistency of Time, Pointing to Objects (POINet)                  |
| Modal    | Modal Prediction                   | Learn representations by encouraging anchor and positive images taken from simultaneous viewpoints or modalities to be close in the embedding space, while distant from negative images taken from different modal data | Siamese, Audio-Visual Correspondence (AVC), Cross-modal Contrastive Learning (CCL), AudioCLIP, Vision-and-Sound-Driven Attention (ViSDA)                                                              |

### GANs

| Pretext Task Type      | Explanation                                                                                                                                       | Model Examples       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| Image reconstruction   | Generate realistic images from a latent space by learning the mapping between the latent space and the image space.                               | GAN, Adversarial AE  |
| Image colorization     | Generate realistic color images from grayscale images by learning the relationship between color channels and luminance information.              | Colorization GAN     |
| Image inpainting       | Fill in missing parts of an image with realistic content by learning the context relationships between different parts of the image.              | Inpainting GAN       |
| Image super-resolution | Generate high-resolution images from low-resolution images by learning the relationship between low-resolution and high-resolution image details. | Super-resolution GAN |

## Backbone Network Architecture
Once the pretext task is defined, the SSL framework can use any architecture to implement the model. In recent years, transformer-based architectures have become popular in SSL due to their ability to model long-range dependencies and capture complex relationships between input examples.
 

| Year | Architectures | Reference                                                                                                | Advantage                                                                                                                               |
| ---- | ------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 2012 | AlexNet       | [[1](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)] | First deep CNN architecture to achieve state-of-the-art results on ImageNet classification.                                             |
| 2014 | VGG           | [[2](https://arxiv.org/abs/1409.1556)]                                                                   | Simpler architecture than previous models, with uniform filter sizes and deep layers.                                                   |
| 2014 | GoogLeNet     | [[3](https://arxiv.org/abs/1409.4842)]                                                                   | Introduces "Inception" modules, which can process multiple filter sizes in parallel.                                                    |
| 2015 | ResNet        | [[4](https://arxiv.org/abs/1512.03385)]                                                                  | Addresses the vanishing gradient problem by using skip connections.                                                                     |
| 2016 | DenseNet      | [[5](https://arxiv.org/abs/1608.06993)]                                                                  | Dense connections between layers allow for more efficient use of parameters.                                                            |
| 2017 | Transformer   | [[6](https://arxiv.org/abs/1706.03762)]                                                                  | First architecture to use self-attention to process sequences, achieving state-of-the-art results in natural language processing tasks. |


## Technology
To optimize the model, SSL frameworks use a variety of technologies and loss functions, including negative sampling, memory banks, contrastive learning, and augmentation strategies. These technologies are designed to reduce overfitting, improve the quality of learned representations, and make the learning process more efficient.

| Stages        | Focus                | Solved what problem       | Explanation                                                               | Model                    |
| ------------- | -------------------- | ------------------------- | ------------------------------------------------------------------------- | ------------------------ |
| Loss Function | Redundancy Reduction | Overfitting               | Reduces the redundancy in the learned features                            | SimCLR, MoCo, BYOL, SwAV |
|               | Memory Bank          | Positive-Negative Pairing | Stores representations of previous batches for sampling negative examples | MoCo, DINO               |
| Sampling      | Negative Sampling    | Overfitting               | Helps the model distinguish between related and unrelated examples        | SimCLR, MoCo, BYOL       |
|               | Unbiased Sampling    | Data bias                 | Reduces the effect of data bias in the learned representations            | SwAV, PCL                |
|              | Hard Negative Sampling   | Hard negative examples | Emphasizes difficult examples to improve the learned representations | PIRL, SeLa |
| Augmentation | Augmentation Strategies  | Data efficiency        | Improves the model's ability to generalize to new data        | SimCLR, BYOL, SwAV, DINO |
|              | Augmentation Regularization | Overfitting          | Encourages the model to learn robust features that are invariant to data augmentations | FixMatch, Noisy Student |
| Architecture | Transformer-based Models | Long-range dependencies | Models the relationships between input examples using self-attention mechanisms | ViT, CCT, DINO |
|              | Contrastive Learning     | Unsupervised learning  | Trains the model to learn a similarity metric between examples without explicit labels | SimCLR, MoCo, BYOL, SwAV, DINO |
|              | Multi-task Learning      | Multi-task learning    | Trains the model to learn multiple tasks simultaneously to improve the learned representations | SimCLR-CLR, MTL, PCL |

## Mentioned Frameworks

### Traditional CV
1.  SimCLR (Simple Contrastive Learning of Representations)
2.  MoCo (Memory-based Contrastive Learning)
3.  BYOL (Bootstrap Your Own Latent)
4.  SwAV (Swapping Assignments between Verbalizations)
5.  DINO (Emerging Properties in Self-Supervised Vision Transformers)
6.  PIRL (Pretraining with Contrastive Learning of Intermediate Representations)
7.  SeLa (Self-Labelling)
8.  CPC (Contrastive Predictive Coding)
9.  CMC (Contrastive Multiview Coding)
10.  PCL (Prototype Conformity Learning)
11.  Noisy Student
12.  FixMatch
13.  SimCLR-CLR (SimCLR for Contrastive Language-Image Retrieval)
14.  MTL (Multi-Task Learning)
15.  ViT (Vision Transformer)
16.  CCT (Consistency-based Semi-supervised Learning for Object Detection)
17.  EsViT (Efficient Self-supervised Vision Transformer)
etc,.


### Remote senisng 
The link need to be updated

| MODAUTY        | APPLICATION                      | METHOD                                                                                                                         |
| -------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Multispectral  | Representation learning          | Vincenzi et al.: colorization as pretext task                                                                                    |
|                |                                  | tile2vec: contrastive learning with triplet loss                                                                                 |
|                |                                  | Jung and Jeon: replace triplet loss of tile2vec to binary cross entropy loss                                                     |
|                |                                  | Stojnic and Risojevic: contrastive multiview coding (CMC)                                                                         |
|                |                                  | CSF: CMC                                                                                                                         |
|                |                                  | Jung et al.: SimCLR with smoothed view                                                                                            |
|                |                                  | GeoMoCo: MoCo + geolocation as pretext task                                                                                       |
|                |                                  | SauMoCo: MoCo with spatial augmentation                                                                                           |
|                |                                  | SeCo: MoCo + seasonal contrast                                                                                                    |
|                |                                  | GeoKR: geo-supervision (landcover map) + teacher-student network                                                                   |
|                | Scene classification             | Lu et al.: AE                                                                                                                    |
|                |                                  | Zhao et al.: rotation as pretext task + classification loss                                                                       |
|                |                                  | Tao et al.: inpainting, relative position and instance discrimination                                                             |
|                |                                  | SGSAGANs: BYOL + GAN                                                                                                              |
|                | Semantic segmentation            | Li et al.: inpainting/rotation as pretext tasks + contrastive learning                                                            |
|                |                                  | Singh et al.: GAN + inpainting as pretext task                                                                                    |
|                |                                  | Li et al.: global and local contrastive learning                                                                                  |
|                | Change detection                 | Zhang et al.: denoising AE                                                                                                       |
|                |                                  | S2-cGAN: conditional GAN discriminator + reconstruction loss                                                                      |
|                |                                  | Dong et al.: GAN discriminator for temporal prediction                                                                            |
|                |                                  | Cai et al.: clustering for hard sample mining                                                                                     |
|                |                                  | Leenstra et al.: triplet loss + binary cross entropy loss                                                                         |
|                |                                  | Chen and Bruzzone: CMC + BYOL                                                                                                     |
|                | Time series classification       | Yuan and Lin: Transformer + temporal pretext task                                                                                 |
|                | Object detection                 | DUPR: patch reidentification + multilevel contrastive loss                                                                        |
|                | Image retrieval                  | Walter et al.: DeepCluster/BiGAN/VAE/colorization                                                                                  |
|                | Depth estimation                 | Madhuanand et al.: AE + multitask prediction + triplet loss                                                                        |
| Hyperspectral  | Image classification             | Mou et al.: AE                                                                                                                   |
|                |                                  | Li et al.: AE + subspace clustering                                                                                               |
|                |                                  | Liu et al.: CMC                                                                                                                  |
|                |                                  | Hu et al.: spatial-spectral Contrastive Clustering (CC)                                                                            |
|                |                                  | Cao et al.: VAE/AAE/PCL                                                                                                           |
|                |                                  | Hu et al.: BYOL + Transformer                                                                                                     |
|                | Unmixing                         | EGU-Net: AE + Siamese weight-sharing                                                                                              |

[References](https://ieeexplore.ieee.org/document/9875399/references#references)

# Dataset

## Fine Tune Dataset

| Dataset    | Short Introduction                                                  | Sample Count | Image Size | Label Count | Reference                                          |
| ---------- | ------------------------------------------------------------------- | ------------ | ---------- | ----------- | -------------------------------------------------- |
| ImageNet   | Large-scale image dataset for object recognition task               | 1.2 million  | 256x256    | 1000        | [[1](http://www.image-net.org/)]                   |
| Places     | Large-scale scene recognition dataset                               | 2.5 million  | 256x256    | 205         | [[2](http://places2.csail.mit.edu/)]               |
| Places365  | Scene recognition dataset with 365 categories                       | 1.8 million  | 256x256    | 365         | [[2](http://places2.csail.mit.edu/)]               |
| SUNCG      | Large-scale indoor scene understanding dataset                      | 45K          | 1280x720   | 40          | [[3](https://sscnet.cs.princeton.edu/)]            |
| MNIST      | Handwritten digits image dataset for classification task            | 70K          | 28x28      | 10          | [[4](http://yann.lecun.com/exdb/mnist/)]           |
| SVHN       | Street View House Numbers image dataset for digit recognition       | 600K         | 32x32      | 10          | [[5](http://ufldl.stanford.edu/housenumbers/)]     |
| CIFAR10    | Object recognition dataset with 10 classes of images                | 60K          | 32x32      | 10          | [[6](https://www.cs.toronto.edu/~kriz/cifar.html)] |
| STL-10     | Object recognition dataset with 10 classes of images                | 130K         | 96x96      | 10          | [[7](https://cs.stanford.edu/~acoates/stl10/)]     |
| PASCAL VOC | Object recognition dataset with 20 classes of images                | 11K          | various    | 20          | [[8](http://host.robots.ox.ac.uk/pascal/VOC/)]     |
| ShapeNet   | Large-scale 3D shape benchmark for object recognition and retrieval | \>51K models |            | 55          | [[9](https://www.shapenet.org/)]                   |
| ModelNet40 | 3D object recognition dataset with 40 categories of objects         | 12K models   |            | 40          | [[10](https://modelnet.cs.princeton.edu/)]         |
| ShapeNet   | Large-scale 3D shape benchmark for object recognition and retrieval | \>51K models |            | 55          | [[9](https://www.shapenet.org/)]                   |


## Earth Observation Datasets

| Dataset    | Short Introduction                                                  | Sample Count | Image Size | Label Count | Spectrum number | Reference                                          |
| ---------- | ------------------------------------------------------------------- | ------------ | ---------- | ----------- | ---------------- | -------------------------------------------------- |
| BigEarthNet | Large-scale Sentinel-2 benchmark archive for land use and land cover | 590,326      | 120x120    | 43          | 13               | https://ieeexplore.ieee.org/abstract/document/8544847 |
| SEN12MS     | A multi-sensor remote sensing dataset for land use classification  | 180,464      | 256x256    | 10          | 1                | https://www.mdpi.com/2072-4292/12/22/3693           |
| So2Sat-LCZ  | A joint classification of land cover and urban morphology          | 660          | 256x256    | 17          | 10               | https://ieeexplore.ieee.org/abstract/document/9418645 |
| EuroSAT     | Land use and land cover classification with Sentinel-2             | 27,000       | 64x64      | 10          | 13               | https://ieeexplore.ieee.org/abstract/document/8202564 |
| NWPU-RESISC | A benchmark for remote sensing image scene classification         | 55,000       | 256x256    | 45          | 3                | https://arxiv.org/abs/2101.04799                     |

# Quality
Community

# Terms
 

| Term                  | Explanation                                                                                                                                                                                                   | Wikipedia Reference |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| pretext task          | A task designed to solve before the actual target task in order to learn **generalizable representations** of inputs at **low and high levels**.                                                              | [Link](https://en.wikipedia.org/wiki/Self-supervised_learning) |
| pseudo-label          | A label generated based on the type of **pre-text task solved by the network**.                                                                                                                               | [Link](https://en.wikipedia.org/wiki/Self-supervised_learning) |
| pre-trained model     | A ConvNet containing **learned data representations/behaviors** that is trained on a **large unlabeled dataset** using **pre-text tasks**.                                                                    | [Link](https://en.wikipedia.org/wiki/Self-supervised_learning) |
| transfer learning     | The process of transferring **pre-trained features** from a pre-trained model to solve **downstream tasks** such as **image classification**, **object detection**, and **image segmentation**.               | [Link](https://en.wikipedia.org/wiki/Transfer_learning) |
| linear classification | A method for evaluating learned representations by training a **linear classifier** on top of a pre-trained ConvNet that is trained on a **large unlabeled dataset** like **ImageNet**.                       | [Link](https://en.wikipedia.org/wiki/Linear_classifier) |
| fine-tuning           | A method for evaluating learned representations by not only replacing and retraining the classifier on top of the ConvNet but also **fine-tuning the pre-trained network's weights** via **backpropagation**. | [Link](https://en.wikipedia.org/wiki/Transfer_learning#Fine-tuning) |
| architecture type     | The quality of visual representations learned through pre-text tasks largely depends on the **ConvNet architecture type** used and the network's **ability to scale with data**.                              | [Link](https://en.wikipedia.org/wiki/Convolutional_neural_network) |
| downstream task       | A task specific to defining what the model is actually supposed to do (**primary task**) while the pretext task is a **secondary task performed to achieve the primary task**.                                | [Link](https://en.wikipedia.org/wiki/Primary_task) |
| contrastive learning  | A method that learns representations by **contrasting similar and dissimilar pairs of samples**.                                                                                                              | [Link](https://en.wikipedia.org/wiki/Contrastive_learning) |
| data augmentation     | A method for creating more training samples by applying transformations such as **rotation, cropping, flipping**, and **color jittering** to existing data.                                                   | [Link](https://en.wikipedia.org/wiki/Data_augmentation) |
