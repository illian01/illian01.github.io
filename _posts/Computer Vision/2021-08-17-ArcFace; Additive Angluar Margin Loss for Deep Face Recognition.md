---
layout: post
title: ArcFace; Additive Angluar Margin Loss for Deep Face Recognition
tag: [ComputerVision, FaceRecognition]
---

{% assign img_path = site.assets_path | append: site.Papers | append: "/" | append: page.title | append: "/" %}

<h3>CVPR, 2018</h3>

---

<h3>1. Introduction</h3>

<p align="center"><img src="{{ img_path }}figure1.png?raw=true" width="60%"></p>

Sphereface, Cosface와 똑같이 softmax의 변형이다.
Sphereface가 $$\cos(m\theta)$$, Cosface가 $$\cos(\theta) - m$$의 형태로 패널티를 준다면 가진다면 Arcface는 $$\cos(\theta + m)$$의 형태를 가진다.

<p align="center"><img src="{{ img_path }}figure2.png?raw=true" width="100%"></p>

---

<h3>2. Proposed approach</h3>

<h4>2.1. ArcFace</h4>

Sphereface, Cosface와 전개방식이 똑같다. 본래 softmax는 다음과 같다.

$$L_{1} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{W^{T}_{y_{i}}x_{i} + b_{y_{i}}}}{\Sigma^{n}_{j=1} e^{W^{T}_{j}x_{i} + b_{j}}}$$

$$W^{T}_{j}x_{i}$$는 $$\lVert W_{j} \rVert \lVert x_{i}\rVert \cos \theta_{j}$$로 쓸 수 있고,
여기서 $$\lVert W_{j} \rVert = 1$$, $$\lVert x_{i} \rVert = s$$로 $$l_{2}$$정규화 한다.

$$L_{2} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s\cos \theta_{y_{i}}}}{e^{s \cos \theta_{y_{i}}} + \Sigma^{n}_{j=1,j \ne y_{i}} e^{s \cos \theta_{j}}}$$

이 식에서 $$\theta_{y_{i}}$$에 additive margin penalty $$m$$ 을 추가한다.

$$L_{3} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s\cos (\theta_{y_{i}} + m)}}{e^{s \cos (\theta_{y_{i}} + m)} + \Sigma^{n}_{j=1,j \ne y_{i}} e^{s \cos \theta_{j}}}$$

아래는 8명에 대해서 각 1,500장씩 샘플링한 후 2차원 특징으로 찍어본 결과라고 한다.

<p align="center"><img src="{{ img_path }}figure3.png?raw=true" width="50%"></p>

<h4>2.2. Comparison with SphereFace and CosFace</h4>

**Numerical similarity.**
Sphereface, Cosface, Arcface 셋 다 margin을 주는 $$m$$값이 있는데, 이 셋을 섞어서 쓸 수도 있다. 
Sphereface의 margin(multiplicative angular margin)을 $$m_{1}$$, Arcface의 margin(additive angular margin)을 $$m_{2}$$,
Cosface의 margin(additive cosine margin)을 $$m_{3}$$라 하자.

$$L_{4} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s(\cos (m_{1}\theta_{y_{i}} + m_{2})-m_{3})}}{e^{s(\cos (m_{1}\theta_{y_{i}} + m_{2})-m_{3})} + \Sigma^{n}_{j=1,j \ne y_{i}} e^{s \cos \theta_{j}}}$$

이걸 combined margin이라고 부를건데, 아래의 오른쪽 그래프가 각 $$m$$이 주어졌을 때 angle에 따른 logit penalty의 정도를 보여준다.

<p align="center"><img src="{{ img_path }}figure4.png?raw=true" width="80%"></p>

왼쪽 그래프는 Arcface를 학습할 때 feature와 weight vector 사이의 각도 분포를 학습 초반, 중반, 후반으로 나눠서 그린 것이다.

**Geometric difference.**
Arcface는 모든 구간에서 일정한 angular margin을 가지기 때문에 우수하다고 말한다. 아래는 이진분류에서의 decision boundary이다.

<p align="center"><img src="{{ img_path }}figure5.png?raw=true" width="80%"></p>

<h4>2.3. Comparison with other losses</h4>

비교군으로 3개의 loss를 제안한다.

**Intra-Loss**
샘플과 GT의 중심을 줄여서 intra-class compactness를 향상시킨다.

$$L_{5} = L_{2} + \frac{1}{\pi N} \sum^{N}_{i=1} \theta_{y_{i}}$$

**Inter-Loss**
샘플과 GT가 아닌 다른 클래스의 중심들과의 거리를 늘려서 inter-class discrepancy를 강화한다.

$$L_{6} = L_{2} - \frac{1}{\pi N(n-1)} \sum^{N}_{i=1} \sum^{n}_{j=1,j \ne y_{i}} \arccos (W^{T}_{y_{i}}, W_{j})$$

**Triplet-loss**
Triplet-loss를 사용한게 FaceNet이 있는데, 여기서는 angular하게 수식을 쓴다.

$$\arccos (x^{pos}_{i}, x_{i}) + m \le \arccos (x^{neg}_{i}, x_{i})$$

---

<h3>3. Experiments</h3>

<h4>3.1. Implementation details</h4>

**Datasets.**

<p align="center"><img src="{{ img_path }}figure6.png?raw=true" width="60%"></p>

이런 데이터셋들을 쓴다. P와 G는 Probe, Gallery를 뜻한다.

**Experimental settings.**
Cosface 논문과 같은 방식으로 정규화된 얼굴을 얻는다(112x112로 얻음). 임베딩 네트워크는 ResNet50과 ResNet100이 사용된다.
마지막 conv 층 뒤에는 BN-Dropout-FC-BN 이 붙어서 특징 벡터를 내놓게 된다.
뒤에서 [training dataset, network structure, loss] 형태로 설정 차이를 표기한다.

feature scale $$s$$는 64, angular margin $$m$$은 0.5를 사용한다. 자세한 설정은 패스.

<h4>3.2. Ablation study on losses</h4>

<p align="center"><img src="{{ img_path }}figure7.png?raw=true" width="60%"></p>

ResNet50과 CASIA 데이터셋으로 학습한 것이다. loss를 다르게 해서 성능이 어떻게 달라지는지 보여준다.

<p align="center"><img src="{{ img_path }}figure8.png?raw=true" width="60%"></p>

[CASIA, ResNet50, loss*]로 학습해서 intra-class와 inter-class 간의 각도 차이를 나타낸 표이다.
NS는 Norm-Softmax를 뜻하고, IntraL, InterL, TripletL은 2.3에 소개된 loss들을 뜻한다.

W-EC는 어떤 특징 벡터와 이에 해당하는 $$W_{j}$$가 있을 때 생기는 각도들의 평균이다.
W-Inter는 $$W_{j}$$들 간의 각도 중 최솟값들의 평균이다.
Intra는 $$x_{i}$$와 특징 중심의 각도 평균, Inter는 다른 특징 중심 중 가장 작은 각도의 평균이다.

1, 2로 구분된 것은 각 CASIA, LFW에 대한 결과임을 뜻한다.

<p align="center"><img src="{{ img_path }}figure9.png?raw=true" width="80%"></p>

LFW에서 positive와 negative 페어들을 모아서 실험했을 때 Arcface가 Triplet보다 잘 구분해낸 다는 것을 말하기 위한 그래프이다.
학습은 [CASIA, ResNet50, loss*]로 했다고 한다.

<p align="center"><img src="{{ img_path }}figure10.png?raw=true" width="60%"></p>

verification 성능

<h4>3.3 Evaluation results</h4>

**Results on LFW, YTF, CALFW, and CPLFW.**

<p align="center"><img src="{{ img_path }}figure11.png?raw=true" width="60%"></p>

verification 성능

<p align="center"><img src="{{ img_path }}figure12.png?raw=true" width="80%"></p>

positive, negative 페어들을 샘플링해서 각도 분포를 나타낸 것이다. [MS1MV2, ResNet100, Arcface]

**Results on MegaFace.**

<p align="center"><img src="{{ img_path }}figure13.png?raw=true" width="60%"></p>

MegaFace Challenge1의 rank-1 face identification과 verification($$TAR@FAR=10^{-6}$$)결과이다.
R은 자체적으로 probe set을 refine한 것이라고 한다.

<p align="center"><img src="{{ img_path }}figure14.png?raw=true" width="80%"></p>

CMC, ROC 곡선으로 Arcface가 Cosface보다 뛰어난 성능을 가졌다는 것을 보인다.

**Results on IJB-B and IJB-C.**

<p align="center"><img src="{{ img_path }}figure15.png?raw=true" width="60%"></p>

verification($$TAR@FAR=10^{-4}$$) 결과이다.

<p align="center"><img src="{{ img_path }}figure16.png?raw=true" width="80%"></p>

IJB-C 의 경우에서는 $$TAR@FAR=10^{-6}$$에서도 뛰어난 성능을 보이고 있다.

**Results on Trillion-Pairs.**

<p align="center"><img src="{{ img_path }}figure17.png?raw=true" width="60%"></p>

identification과 verification 결과

**Results on iQIYI-VID.**

<p align="center"><img src="{{ img_path }}figure18.png?raw=true" width="60%"></p>

---

<h3>4. Conclusions</h3>

pass

---

<h3>5. Appendix</h3>

pass

