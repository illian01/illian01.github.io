---
layout: post
title: Linkage Based Face Clustering via Graph Convolution Network
tag: [ComputerVision, FaceClustering]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" %}

<h3>CVPR, 2019</h3>

---

<h3>1. Introduction</h3>

<p align="center"><img src="{{ img_path }}figure1.png?raw=true" width="50%"></p>

---

<h3>2. Related work</h3>

pass

---

<h3>3. Proposed approach</h3>

<h4>3.1. Overview</h4>

**Problem definition.** 얼굴 이미지의 특징들이 $$X \in \left[ x_{1}, ..., x_{N} \right]^{T}$$로 주어진다.
$$N$$은 이미지의 수, $$D$$는 특징의 차원이다. 각 얼굴 이미지는 label이 존재하고, 클러스터링을 통해 pseudo label을 부여한다.

**Motivation.** 데이터셋에서 k-NN을 구하고, 다른 레이블인 데이터끼리의 link를 제거한다.
이렇게 얻어진 링크들로 클러스터링을 했을 때 얻는 점수가 upper bound라고 한다면, 꽤 합리적인 목표가 된다.

<p align="center"><img src="{{ img_path }}figure2.png?raw=true" width="80%"></p>

IJB-B-512 데이터셋에 대해서 k-NN을 적용하고, 다른 레이블끼리의 link들을 제거한 뒤, connected component들을 클러스터 결과로 사용한 결과이다.

**Pipeline.** 

- 모든 instance들을 pivot으로 사용한다. 각 pivot에 대해서 Instance Pivot Subgraph (IPS)를 구성해서 사용한다.
- IPS가 입력 데이터로, graph convolution network를 적용한다. 이 네트워크는 pivot과 이웃 사이의 link를 예측한다.
- 네트워크를 통해 weighted edge들을 얻었으면, 이들을 통합해 클러스터를 구축한다.

<p align="center"><img src="{{ img_path }}figure3.png?raw=true" width="60%"></p>

<h4>3.2. Construction of instance pivot subgraph</h4>

**Step 1: Node discovery.** pivot $$p$$에 대해서 주변의 node들을 IPS에 추가한다.
pivot을 기준으로 $$h$$-hop까지의 이웃들이 추가되는데, 각 $$i$$-hop에서는 $$k_{i}$$개의 최근접 이웃이 선택된다.
$$h=3$$, $$k_{1}=8$$, $$k_{2}=4$$, $$k_{3}=2$$라고 했을 때, 1-hop으로 pivot의 최근접이웃 8개, 각 1-hop node들의 최근접이웃 4개(총 32개), 마지막으로 각 2-htop node들의 최근접 이웃 2개(총 64)가 추가되어 IPS에 총합 104개의 node가 포함된다.

**Step 2: Node feature normalization.** pivot $$p$$와 주변의 node set $$V_{p}$$를 얻었다. 
pivot정보를 IPS에 encode해주기 위해, pivot 특징 $$x_{p}$$를 IPS의 모든 node에서 뺄셈한다.

$$\mathcal{F}_{p} = \left[ ...,x_{q}-x_{p},...\right]^{T}, \mbox{for all } q\in V_{p}$$

$$\mathcal{F}_{p} \in \mathbb{R} ^{ \mid V_{p} \mid \times D} $$ 이다.

<p align="center"><img src="{{ img_path }}figure4.png?raw=true" width="100%"></p>

**Step 3: Adding edges among nodes.** node가 구성되었으니 edge를 추가하면 된다.
각 노드 $$q \in V_{p}$$에 대해서 전체 데이터셋과 비교했을 때, $$u$$개의 최근접 이웃을 구할 수 있다.
이 $$u$$개의 최근접 이웃 중 한 노드 $$r$$이 IPS에 존재한다면, edge set $$E_{p}$$에 $$(q,r)$$을 추가한다.
이 edge들을 인접행렬 $$A_{p} \in \mathbb{R}^{\mid V_{p} \mid \times \mid V_{p} \mid}$$로 나타낼 수 있다.

<h4>3.3. Graph convolutions on IPS</h4>

GCN(Graph Convolution Network)은 입력으로 특징 행렬 $$X$$와 인접행렬 $$A$$를 받고, 변형이 가해진 특징 행렬 $$Y$$를 출력한다.

$$Y = \sigma ( [ X \parallel GX ] W )$$

$$X \in \mathbb{R}^{N \times d_{in}} $$, $$Y \in \mathbb{R}^{N \times d_{out} }$$이고, 
여기서 $$N$$은 노드의 수, $$d_{in}$$과 $$d_{out}$$은 특징의 입출력 차원 수 이다.
$$G=g(X,A)$$로 연산되는 aggregation matrix로, $$N \times N$$의 크기를 가지며 각 행의 합이 1인 특징을 가진다.
$$\parallel$$ 연산은 특징의 차원 방향으로 concat하는 연산이다. 
즉, $$X \in \mathbb{R}^{N \times d_{in}}$$와 $$GX \in \mathbb{R}^{N \times d_{in}}$$을 concat하므로,
$$[ X \parallel GX ] \in \mathbb{R}^{N \times d_{2in}}$$가 된다. 
$$W$$는 학습하는 weight로, $$2d_{in} \times d_{out}$$을 크기를 가지고, $$\sigma$$는 ReLU이다.

Aggregation matrix를 만드는 $$g( \cdot )$$함수를 Mean aggregation, Weighted aggregation, Attention aggregation 세 가지로 실험해 봤는데,
Mean aggregation 방식이 성능이 가장 좋았다고 한다.

$$G = \Lambda^{-\frac{1}{2}} A \Lambda^{-\frac{1}{2}}, \Lambda_{ii} = \Sigma_{j} A_{ij} $$

GCN은 IPS를 입력으로 받아서 pivot과 IPS의 각 노드 사이의 link를 예측한다. 따라서, softmax와 cross-entropy loss가 적용된다.
IPS를 구성할 때, pivot으로부터 $$h$$-hop 까지의 node들을 가져와 구성했는데, 
실제로 link를 예측하고, loss와 gradient를 계산하는 것은 1-hop인 node에 대해서만 시행한다.
아래 그림은 GCN을 통과할 때마다 특징 노드들이 비슷한 것끼리 모인다는 것을 표현한 것이다.

<p align="center"><img src="{{ img_path }}figure5.png?raw=true" width="60%"></p>

<h4>3.4. Link merging</h4>

모든 instance들을 pivot으로 IPS를 추출 후, GCN을 통과시킨다면, 각 instance와 그의 최근접 이웃들 사이의 edge score를 얻을 수 있다.
이 edge score들을 threshold로 낮은 것들을 버리고 BFS를 적용해도 되나, 이 논문에서는 
*"X. Zhan. Z. Liu, J. Yan, D. Lin, and C. Change Loy. Consessus-driven propagation in massive unlabeled data for face recognition. In ECCV, 2018."*
에서 사용한 방식을 따른다고 한다.

---

<h3>4. Experiment</h3>

<h4>4.1. Evaluation metrics and datasets</h4>

NMI(Normalized Mutual Information), BCubed F-measure가 성능 지표로 사용된다.
얼굴 특징 추출 모델로는 MS-Celeb-1M과 VGGFace2으로 학습된 ArcFace 모델을 사용한다.
GCN 모델을 학습시키기 위한 데이터셋으로는 CASIA 데이터셋에서 5K의 인물과 200K의 이미지가 포함되도록 랜덤 샘플링된 서브셋을 사용한다.
마지막으로, GCN 성능 테스트를 위한 데이터셋으로 IJB-B 데이터셋을 사용한다.

<h4>4.2. Parameter selection</h4>

$$h=2$$를 사용한다.
학습에는 $$k_{1}=200$$, $$k_{2}=10$$, $$u=10$$을 사용하고, 테스트에는 $$k_{1}=80$$, $$k_{2}=5$$, $$u=5$$를 사용한다.
모든 값은 실험적으로 얻어낸 값이다.

<p align="center"><img src="{{ img_path }}figure6.png?raw=true" width="80%"></p>

<h4>4.3. Evaluation</h4>

**Comparing different aggregation methods.** GCN 층의 aggregation 방식의 차이에 따라 GCN-M, GCN-W, GCN-A라고 표기한다.
각각 Mean aggregation, Weighted aggregation, Attention aggregation을 의미한다.

**Comparison with baseline methods.** K-means, Spectral clustering, AHC, AP, DBSCAN

**Comparison with state-of-the-art.** ARO, PAHC, ConPaC, DDC

<p align="center"><img src="{{ img_path }}figure7.png?raw=true" width="80%"></p>

**Different face representation.** ArcFace 특징이 아닌 다른 얼굴 특징에 대해서도 뛰어난 성능을 내는지 보인다.
얼굴 특징 추출 모델로 ResNet-50 + Softmax Loss를 MS1M 데이터셋으로 학습시킨다. 
아래 표는 IJB 데이터셋으로 클러스터링 실험을 했을 때의 결과이며, 여전히 가장 높은 성능을 보이는 것을 알 수 있다.

<p align="center"><img src="{{ img_path }}figure8.png?raw=true" width="80%"></p>

**Singleton clusters.**
Singleton cluster는 node가 1개인 클러스터를 의미한다. 
각 알고리즘의 하이퍼파라미터를 조정하며 다양한 비율의 singleton cluster가 나오도록 조정하고, 
이 singleton cluster를 제외하고 성능 지표를 측정한다.
일반적으로 singleton cluster는 저화질, 어려운 샘플, 잘못된 디텍션된 얼굴 등인 경우가 많은데, 이들에 강건한 알고리즘이라는 것을 보이기 위한 것이다.
사용한 데이터셋은 IJB-B-512이다.

<p align="center"><img src="{{ img_path }}figure9.png?raw=true" width="80%"></p>

**Scalability and efficiency.** 

IPS 구축을 위한 nearest neighbor 검색은 $$O(n \log n)$$ 시간이 걸리는 ANN(Approximate Nearest Neighbor)를 채용한다.
다양한 $$k$$값에 따른 모델 예측 시간과 성능을 비교한다. 
데이터셋은 IJB-B-1845을 사용하되, Megaface 데이터셋을 distractor로 사용한다.
성능을 측정할 때는 distractor는 제외하고 측정하게 된다. 장비는 한 개의 Titan Xp GPU를 사용했다고 한다.

<p align="center"><img src="{{ img_path }}figure10.png?raw=true" width="80%"></p>

<h4>4.4. Multi-view extension</h4>

이미지와 오디오 특징을 둘 다 사용한다. 각 데이터를 CNN으로 통과해서 얻은 특징을 concat해 사용한다.
데이터셋은 VoxCeleb2을 사용하며, 2048 명의 인물을 겹치지 않게 추출해서 테스트셋으로 사용한다.

<p align="center"><img src="{{ img_path }}figure11.png?raw=true" width="80%"></p>

테스트셋의 일부분 512 인물에 대한 결과

<p align="center"><img src="{{ img_path }}figure12.png?raw=true" width="80%"></p>

테스트셋 전체인 2048 인물에 대한 결과

---

<h3>5. Conclusion</h3>

pass

---

<h3>6. Acknowledgement</h3>

pass

