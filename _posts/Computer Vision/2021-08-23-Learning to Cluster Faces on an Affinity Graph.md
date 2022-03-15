---
layout: post
title: Learning to Cluster Faces on an Affinity Graph
tag: [ComputerVision, Clustering]
---

{% assign img_path = site.assets_path | append: site.Papers | append: "/" | append: page.title | append: "/" %}

<h3>CVPR, 2019</h3>

---

<h3>1. Introduction</h3>

얼굴 특징으로 그래프를 구축하고 GCN으로 학습한다.
GCN 클러스터링은 Mask R-CNN의 파이프라인을 본따서 proposal 생성, 그 중 점수가 높은 것 선정 후 refine하는 단계를 거친다.

<p align="center"><img src="{{ img_path }}figure1.png?raw=true" width="60%"></p>

---

<h3>2. Related work</h3>

pass

---

<h3>3. Methodology</h3>

얼굴 데이터셋을 CNN에 통과시켜 특징 셋 $$\mathcal{D} = \{ f_{i} \} ^{N}_{i=1}$$를 얻는다.
$$f_{i}$$는 $$d$$차원 벡터이다.
각 특징들을 vertex로 두고, cosine similarity와 K-NN으로 affinity graph $$\mathcal{G} = (\mathcal{V}, \mathcal{E})$$를 얻는다.
전체 데이터셋에 대해서 그래프를 구축하기 때문에 엄청나게 큰 그래프를 가지게 된다.

<h4>3.1. Framework overview</h4>

<p align="center"><img src="{{ img_path }}figure2.png?raw=true" width="100%"></p>

중요하게 proposal generator, GCN-D, GCN-S 3개로 구성된다.
proposal generator는 affinity graph에서 클러스터 일 것 같은 sub-graph들을 추출하고,
이 sub-graph(proposal)들이 GCN-D, GCN-S에 들어가게 된다.

GCN-D는 dectection이라고 이름 붙여져 있는데, 하는 일은 proposal에 대해서 얼마나 cluster 같은지 점수를 맥이는 역할을 한다.
여기서 높은 점수를 얻은 proposal이 있다면 GCN-S로 넘어가게 되고, GCN-S는 segmentation 작업을 한다.
각 vertex에 대해서 noise 확률을 얻어내고 이들을 쳐내서 refine하면 좀 더 나은 클러스터 결과를 얻을 수 있다는 것이다.

<h4>3.2. Cluster proposals</h4>

cluster proposal set을 $$\mathcal{P} = \{ \mathcal{P}_{i} \}^{N_{p}}_{i=1}$$라 한다.
proposal들은 super-vertex들을 기반으로 생성되는데 super-vertex set을 $$\mathcal{S} = \{ \mathcal{S}_{i} \}^{N_{s}}_{i=1}$$라 한다.

**Super-Vertex.**
<p align="center"><img src="{{ img_path }}figure3.png?raw=true" width="60%"></p>

super-vertex는 대충 super-pixel의 vertex 버전으로 표현한 것 같다.
비슷한 vertex를 묶어서 하나의 super-vertex가 된다.
affinity value가 너무 작은 경우를 쳐내기 위해 threshold $$e_{\tau}$$가 쓰인다.
또한 super-vertex가 너무 커지지 않게 최대 $$s_{max}$$ 크기로 제한한다.

보통 1M개 vertex들이 있으면 50K의 super-vertex가 생기고 약 20개의 vertex를 가진다고 한다.

**Proposal generation.**
<p align="center"><img src="{{ img_path }}figure4.png?raw=true" width="60%"></p>

super-vertex를 하나의 vertex로 보고 알고리즘1 을 여러번 거쳐가는 방식으로 proposal을 생성한다.
즉, super-vertex를 작은 클러스터로 보고 클러스터 중심간의 affinity를 기준으로 합쳐나간다는 뜻이 되겠다.

<h4>3.3. Cluster detection</h4>

proposal $$\mathcal{P}$$에 대해서 두 가지 metric이 주어진다.

$$
\begin{align*}
IoU(\mathcal{P}) = \frac{\lvert \mathcal{P} \cap \hat{\mathcal{P}} \rvert}{\lvert \mathcal{P} \cup \hat{\mathcal{P}} \rvert}, 
&& 
IoP(\mathcal{P}) = \frac{\lvert \mathcal{P} \cap \hat{\mathcal{P}} \rvert}{\mathcal{\lvert P \rvert}}
\end{align*}
$$

$$\hat{\mathcal{P}}$$는 ground-truth로 레이블 $$l(\mathcal{P})$$ 인 vertex로만 구성된 것을 뜻하며,
$$l(\mathcal{P})$$는 $$\mathcal{P}$$의 majority label을 의미한다.
$$IoU$$는 $$\mathcal{P}$$가 $$\hat{\mathcal{P}}$$와 얼마나 유사한지,
$$IoP$$는 $$\mathcal{P}$$에서 $$l(\mathcal{P})$$의 비율(purity)을 말해준다.

**Design of GCN-D.**
cluster proposal $$\mathcal{P}_{i}$$, $$l$$번 층의 입력 특징 행렬 $$F_{l}(\mathcal{P}_{i})$$, sub-matrix의 affinity matrix $$A(\mathcal{P}_{i})$$가 주어질 때,
$$l$$번 층의 연산은 다음과 같다.

$$F_{l+1}(\mathcal{P}_{i}) = \sigma (\tilde{D}(\mathcal{P}_{i})^{-1} (A(\mathcal{P}_{i}) + I) F_{l}(\mathcal{P}_{i}) W_{l})$$

$$\tilde{D} = \Sigma_{j} \tilde{A}_{ij}(\mathcal{P}_{i})$$는 차수행렬이고, $$\sigma$$는 $$ReLU$$ 활성함수이다.

graph conv들의 끝에 max pooling이 진행되고, $$IoU$$와 $$IoP$$를 예측하기 위한 두 층의 FC가 붙는다.

**Training and inference.**
$$IoU, IoP$$에 MSE(Mean Square Error) loss로 학습한다.

<h4>3.4. Cluster segmentation</h4>

GCN-D를 통과한 proposal들에 대해서 outlier들을 제거하기 위한 네트워크이다.

**Design of GCN-S.**
GCN-S는 proposal $$\mathcal{P}$$의 각 vertex $$v$$에 대해서 클러스터에 속하는지 아닌지 분류한다.

**Identifying outliers.**
proposal에 어떤게 positive고 negative인지는 가장 많은 클래스로 정해도 되긴 하는데, 2개 클래스가 반반 정도 먹고있는 것이 나오면 정하기가 곤란해진다.
그래서 random seed를 두고 하나의 클래스를 positive로, 나머지를 negative로 두는 것을 proposal마다 여러번 진행해서 하나의 proposal마다 다수의 샘플을 만든다고 한다. 
이 때, random seed의 특징에는 1 값을, 나머지 특징에는 0 값을 concat해줌으로써 어떤 특징을 기준으로 positive, negative를 구분할지 인코딩 해준다.

**Training and inference.**
위의 과정에서 positive와 negative vertex가 구분된 proposal들을 얻을 수 있다. 
이를 vertex 마다 BCE(Binary Cross Entropy) loss로 학습한다. 

추론 단계에서도 학습 샘플을 만드느 것과 같이, 하나의 proposal에서 여러 hypotheses를 얻어서 사용한다.
이 샘플들을 GCN-S에 입력으로 주고, positive가 proposal에서 가지는 비율이 threshold 보다 높은 proposal들만 클러스터로 인정된다.
논문에서는 이 threshold를 0.5로 사용했다고 한다.

학습에 사용하는 proposal은 $$0.3 \le IoP \le 0.7$$인 것만 사용한다.
GCN-S의 학습의 경우에는 너무 한쪽으로 치우쳐져 있으면 어려운 샘플이 되기 때문이다.

<h4>3.5. De-Overlapping</h4>

<p align="center"><img src="{{ img_path }}figure5.png?raw=true" width="60%"></p>

여전히 많은 proposal들이 겹쳐있는 형태로 남아있을 수 있기 때문에 de-overlapping 과정을 거친다.
먼저 IoU 점수 내림차순으로 proposal들을 정렬한다.
그 후, 차례대로 proposal에서 이전 proposal에 나왔던 vertex들을 지우는 과정을 거친다.
시간복잡도가 $$O(N)$$이라서 빠르다는 장점이 있다.

---

<h3>4. Experiments</h3>

<h4>4.1. Experimental settings</h4>

**Training set.**
MS-Celeb-1M을 한번 손봐서 5.8M 이미지와 86K 클래스를 가지도록 만든 뒤, 10등분해서 1개는 labeled로 쓰고, 9개는 unlabeled로 쓴다고 한다.
클래스가 여러 등분에 동시에 존재하도록 나누지 않는 것 같다. 즉, 각 등분이 580K 이미지에 8.6K 클래스를 가진다.
YouTube Faces 데이터셋은 3,425 개의 동영상, 프레임으로 144,882개의 프레임이 있다.
이 중 14,653 프레임, 159 클래스를 training으로 사용하며, 140,629 프레임, 1,436 클래스를 testing으로 사용한다.

**Testing set.**
MegaFace와 IJB-A가 쓰인다.

근데 어차피 각 데이터셋 쪼개서 실험하는거 같은데 왜 이렇게 쓴거지

**Metrics.**
face clustering에서는 pairwise recall, precision을 재고 이 둘의 조화평균인 F-score까지 사용한다.
face identification에서는 top-1 hit rate를, face verification에서는 FPR 0.001에서의 TPR을 본다.

**Implementation details.**
GCN의 hidden layer는 두 층 이라고 한다. SGD에 lr=0.01, momentum이 적용된다.
알고리즘1에서 $$e_{\tau} \in \{ 0.6, 0.65, 0.7, 0.75 \}, s_{max}=300$$이다.

<h4>4.2. Method comparison</h4>

<h4>4.2.1. Face clustering</h4>

1. K-means
2. DBSCAN
3. HAC
4. Approximate rank order
5. CDP
6. GCN-D
7. GCN-D + GCN-S

**Results.**
<p align="center"><img src="{{ img_path }}figure6.png?raw=true" width="80%"></p>

MS-Celeb-1M을 10토막 낸 것 중 1개를 가지고 실험한 결과이다. 

<p align="center"><img src="{{ img_path }}figure7.png?raw=true" width="80%"></p>

YouTube Faces에 실험한 결과이다. 다양한 분포를 가진 데이터가 들어와도 robust하다는 것을 보여준다고 한다.

**Runtime analysis.**
2200초가 150K개 proposal을 만드는데 1000초, GCN-D, GCN-S를 통과하는데 각 1000초, 200초가 걸렸다고 한다.
batch를 32로 돌렸다고 하고, GPU말고 CPU에서 돌려도 3700초가 걸렸다고 한다.
GCN의 연산 대부분이 sparse matrix multiplication이라 GPU의 기능을 최대로 끌어올리지 못하는 것이 이유라고 한다.

<h4>4.2.2. Face recognition</h4>

데이터셋을 10개로 쪼갠다음 1개를 labeled, 9개를 unlabeled로 써서 face recognition 모델을 다음과 같이 학습한다.

1. labeled set으로 초기 모델을 학습한다.
2. labeled set과 1 에서 학습한 모델로 clustering 모델을 학습한다.
3. 클러스터링 모델로 unlabeled data에 pseudo-label을 붙인다.
4. pseudo-label과 labeled set을 합쳐서 전체 데이터셋을 만들고 이걸로 recognition model을 학습한다.

<p align="center"><img src="{{ img_path }}figure8.png?raw=true" width="80%"></p>

전부 labeled로 학습한 것이 upper bound 이다.

<h4>4.3. Ablation study</h4>

<h4>4.3.1. Proposal strategies</h4>
<p align="center"><img src="{{ img_path }}figure9.png?raw=true" width="60%"></p>

k-NN의 k를 $$K=80$$으로 고정시키고 $$I, e_{\tau}, s_{max}$$를 다르게 해서 proposal 수를 다르게 해서 얻은 그래프라고 한다.

<h4>4.3.2. Design choice of GCN-D</h4>
<p align="center"><img src="{{ img_path }}figure10.png?raw=true" width="60%"></p>

max pooling이 좋았고, feature 정보는 필요하고, 너무 깊은 GCN은 안좋다더라 라는게 결론이라 한다.

<h4>4.3.3. GCN-S</h4>
<p align="center"><img src="{{ img_path }}figure11.png?raw=true" width="60%"></p>

GCN-S를 cluster de-noising 처럼 학습했기 때문에 다른 알고리즘에도 붙여쓸 수 있다. 다른 것에 써도 성능을 부스팅 하는 것을 알 수 있다.

<h4>4.3.4. Post-process strategies</h4>
<p align="center"><img src="{{ img_path }}figure12.png?raw=true" width="60%"></p>

NMS가 보통 post-processing으로 많이 쓰는데, 시간도 오래걸리고 다른 proposal을 버리는 방법을 쓰는 것보다 제안한 de-overlapping을 하는 것이 낫다 라고 한다.

---

<h3>5. Conclusions</h3>

내가 잘못 읽은 건가 좀 생략된 내용이 많은 것 같다. 특징 뽑는 네트워크도 ArcFace를 쓴 것 같은데 음..