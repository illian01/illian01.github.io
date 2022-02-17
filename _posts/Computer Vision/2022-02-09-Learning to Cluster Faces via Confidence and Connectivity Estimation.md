---
layout: post
title: Learning to Cluster Faces via Confidence and Connectivity Estimation
tag: [ComputerVision, FaceClustering]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" %} 

<h3>CVPR, 2020</h3>

---

<h3>1. Introduction</h3>

클러스터링 과정을 GCN-V와 GCN-E라는 두 네트워크로 분리한다.
GCN-V는 vertex의 confidence를 예측하고, GCN-E는 edge를 예측한다.

<p align="center"><img src="{{ img_path }}figure1.png?raw=true" width="50%"></p>

---

<h3>2. Related work</h3>

pass

---

<h3>3. Methodology</h3>

<h4>3.1. Framework overview</h4>
얼굴 이미지 데이터셋을 학습된 CNN에 통과시켜 얻은 특징 셋을 $$\mathcal{F}= \{ \mathrm{f}_{i} \}^{N}_{i=1}$$로 표기한다.
$$\mathrm{f}_{i} \in \mathbb{R}^{D}$$이며, $$N$$은 데이터의 수, $$D$$는 특징의 차원이다.
샘플 $$i$$와 $$j$$의 affinity를 $$a_{i,j}$$로 표기하고, 이는 $$\mathrm{f}_{i}$$와 $$\mathrm{f}_{j}$$의 cosine similarity로 계산한다.
이 affinity들을 가지고 kNN affinity graph $$\mathcal{G}=(\mathcal{V}, \mathcal{E})$$를 구한다.
이미지의 특징 셋이 $$\mathcal{V}$$가 되고, 각 vertex에서 뻗어지는 k edge들이 $$\mathcal{E}$$에 포함된다.
구성된 그래프의 특징 행렬은 $$\mathrm{F} \in \mathbb{R}^{N \times D}$$로 표기하고, 
유사도 행렬은 대칭행렬로 구성해 $$\mathrm{A} \in \mathbb{R}^{N \times N}$$로 표기한다.
논문엔 adjacency라고 써져있는데, affinity가 맞는 듯 하다.

클러스터링을 2개의 sub-problem으로 구분한다.
첫 번째는 GCN-V로, vertex들에 대해서 confidence를 예측한다.
confidence는 이웃 vertex들이 얼마나 같은 클래스들로 밀집되어 있는지를 나타낸다.
두 번째는 GCN-E로, vertex 사이의 edge를 예측한다. 같은 클래스라면 edge를 형성하고, 다른 클래스라면 형성하지 않는다.

<p align="center"><img src="{{ img_path }}figure2.png?raw=true" width="80%"></p>

<h4>3.2. Confidence estimator</h4>
GCN-V는 각 vertex마다 confidence를 예측한다. confidence $$c_{i}$$는 다음과 같이 정의된다.

$$c_{i} = \frac{1}{\lvert \mathcal{N}_{i} \rvert} \sum_{v_{i} \in \mathcal{N_{i}}} \left( \doubleone_{y_{j}=y_{i}} -\doubleone_{y_{j} \ne y_{i}} \right) \cdot a_{i,j}$$

$$\mathcal{N}_{i}$$는 $$v_{i}$$의 neighbor, $$y_{i}$$는 $$v_{i}$$의 ground-truth, $$a_{i,j}$$는 $$v_{i}$$와 $$v_{j}$$사이의 affinity이다.
그리고, $$\doubleone$$은 밑의 조건이 참일 때 1, 거짓일 때 0이 되는 값이다.
이에 따르면, 한 노드의 이웃이 같은 클래스들로 밀집되어 있다면 높은 confidence를 가질 것이고, 밀집 정도가 낮거나 다른 클래스들이 많다면 상대적으로 낮은 confidence를 가지게 될 것이다.

**Design of confidence estimator.**
그래프 구조를 학습하기 위해 graph convolution network를 사용한다.
$$l$$번째 층의 graph convolution은 다음과 같은 연산을 한다.

$$\mathrm{F}_{l+1} = \sigma \left( g \left( \tilde{\mathrm{A}}, \mathrm{F}_{l} \right) \mathrm{W}_{l} \right)$$

$$\tilde{\mathrm{A}}= \tilde{\mathrm{D}}^{-1}(A+I)$$, $$\tilde{\mathrm{D}}_{ii} = \Sigma_{j} (A+I)$$이며,
$$\sigma$$는 ReLU 활성함수이다.
$$g(\cdot, \cdot)$$은 neighborhood aggregation으로, 다음과 같은 연산이다.

$$g\left( \tilde{\mathrm{A}}, \mathrm{F}_{l} \right) = \left[ (\mathrm{F}_{l})^{\top} , (\tilde{\mathrm{A}} \mathrm{F}_{l})^{\top} \right]^{\top} $$

기존의 특징에 이웃들을 aggregation한 특징을 concat하는 것이 좀 더 나은 성능을 보인다고 한다.
GCN을 통과한 후, fully connected layer를 통해 confidence를 예측한다.

$$v_{i}$$의 confidence score는 $$c^{'}_{i}$$가 된다.

**Training and inference.**
GCN-V는 MSE(Mean Squared Error)를 통해 confidence score를 학습한다.

$$\mathcal{L}_{V} = \frac{1}{N} \sum^{N}_{i=1} \lvert c_{i} - c^{'}_{i} \rvert^{2}$$

예측된 confidence는 GCN-E의 입력 데이터를 만들 때와 마지막 클러스터 결과를 생성할 때 사용된다.

**Complexity analysis.**
pass

<h4>3.3. Connectivity estimator</h4>
GCN-E는 vertex 사이의 edge를 예측하되, 이웃 중 confidence가 자기보다 높은 vertex로 edge를 예측한다.

**Candidate set.**
$$v_{i}$$에 대해서 edge 예측을 진행할 candidate set $$\mathcal{S}_{i}$$를 구한다.

$$\mathcal{S}_{i} = \left\{ v_{j} \vert c^{'}_{j} > c^{'}_{i}, v_{j} \in \mathcal{N}_{i} \right\}$$

$$\mathcal{S}_{i}$$에는 $$v_{i}$$의 confidence $$c^{'}_{i}$$보다 높은 confidence를 가지는 $$v_{j}$$만 포함된다.

**Design of connectivity estimator.**
GCN-E도 GCN-V와 같은 graph convolution 연산을 한다. 차이점이라면 입력 값의 차이가 있다.
전체 그래프 대신 subgraph $$\mathcal{G}(\mathcal{S}_{i})$$가 들어간다.
subgraph $$\mathcal{G}(\mathcal{S}_{i})$$의 affinity matrix를 $$\mathrm{A}(\mathcal{S}_{i})$$, feature matrix를 $$\mathrm{F}(\mathcal{S}_{i})$$로 쓴다.
여기에 subgraph $$\mathcal{S}_{i}$$와 $$v_{i}$$의 관계를 인코딩 해주기 위해 $$\mathrm{F}(\mathcal{S}_{i})$$의 각 행에 $$\mathrm{f}_{i}$$를 뺄셈 연산한다.
이를 $$\bar{\mathrm{F}}(\mathcal{S}_{i})$$로 쓴다.

$$\bar{\mathrm{F}}_{l+1} = \sigma \left( g(\tilde{\mathrm{A}}(\mathcal{C}_{i}), \bar{\mathrm{F}_{l}(\mathcal{C}_{i})}) \mathrm{W}^{'}_{l}  \right)$$

GCN-E의 끝단에 fc layer가 붙어서 $$v_{i}$$와 $$v_{j}$$의 관계 $$r_{i,j}$$를 예측한다.
예측값은 $$r^{'}_{i,j}$$로 표기한다.

**Training and inference.**
$$r_{i,j}$$는 $$v_{i}, v_{j}$$가 같은 클래스일 때 1, 아니면 0인 값이다.

$$r_{i,j}=
\begin{cases}
1, & y_{i} = y_{j} \\
0, & y_{i} \ne y_{j}
\end{cases}
, v_{j} \in \mathcal{S}_{i}
$$

MSE를 최소화 하는 것으로 학습한다.

$$\mathcal{L}_{E} (\mathcal{C}_{i}) = \sum_{v_{j} \in \mathcal{C}_{i}} \lvert r_{i,j} - r^{'}_{i,j} \rvert^{2}$$

학습과 추론 속도 향상을 위해 모든 vertex에 대해서 GCN-E 과정을 거치는 것이 아니라 $$\rho$$ 비율의 vertex만 추출해 사용한다.
vertex를 추출하는 순서는 높은 confidence를 가진 vertex부터 사용한다.
나머지 GCN-E에 사용하지 않는 vertex들은 candidate set 내의 M-최근접 이웃으로 바로 간선을 연결하는 방식을 사용한다.
경험적으로 $$M=1, \rho=10%$$ 일 때, 충분한 성능이 나왔다고 한다.

**Complexity analysis.**
pass

---

<h3>4. Experiments</h3>

<h4>4.1. Experimental settings</h4>

**Face clustering.**
MS-Celeb-1MV2를 사용한다. 이를 클래스가 겹치지 않게 10등분해서 1개 파트를 labeled, 9개 파트를 unlabeled로 사용한다.

**Fashion clustering.**
3,997개의 카테고리에 대해서 25,752장의 이미지를 train set으로, 
3,984개의 카테고리에 대해서 26,960장의 이미지를 test set으로 사용한다. 
train과 test 사이에 겹치는 카테고리는 없다.

**Face recognition.** MegaFace 데이터셋을 사용한다.

**Metrics.**
클러스터링 성능 측정은 Pairwise F-score와 BCubed F-score를 사용한다.
MegaFace의 face identification은 top-1 hit rate를 사용한다.

**Implementation details.**
k-NN 그래프를 위한 $$k$$ 값으로 MS1M 은 80, Deep-Fasion은 5를 사용한다.
GCN-V는 1층, GCN-E는 4층의 GCN으로 구성된다.
optimizer는 SGD + momentum, lr=0.1, weight_decay=1e-5이다.
부정확한 간선이 포함되지 않도록 threshold $$\tau=0.8$$로 두고 낮은 값을 가지는 것은 잘라낸다.

<h4>4.2. Method comparison</h4>

<h4>4.2.1. Face clustering</h4>
K-means, HAC, DBSCAN, MeanShift, Spectral, ARO, CDP, L-GCN, LTC, GCN-V, GCN-VE

**Results.** MS1MV2 labeled에 학습, unlabeled에 성능 측정

<p align="center"><img src="{{ img_path }}figure3.png?raw=true" width="80%"></p>

Deep-Fashion에 대한 결과

<p align="center"><img src="{{ img_path }}figure4.png?raw=true" width="50%"></p>

**Runtime analysis.**
pass

<h4>4.2.2. Face recognition</h4>

MS1MV2를 사용해서 다음과 같이 face recognition model을 만든다.

1. labeled set으로 face recognition model을 학습한다.
2. 학습돤 face recognition model과 labeled set으로 face clustering model을 학습한다.
3. 학습된 face clustering model로 unlabeled set에 pseudo label을 부여한다.
4. labeled set과 pseudo lebeled set으로 face recognition model을 학습한다.

이전 work에서는 pseudo label을 부여할 때, unlabeled 비율에 따라 클러스터링을 여러번 진행했으나,
이 논문에서는 unlabeled 전체를 한번에 클러스터링 한 후, 여러 비율에 맞게 split해서 사용했다고 한다.

<p align="center"><img src="{{ img_path }}figure5.png?raw=true" width="50%"></p>

<h4>4.3. Ablation study</h4>

pass

---

<h3>5. Conclusion</h3>

pass
