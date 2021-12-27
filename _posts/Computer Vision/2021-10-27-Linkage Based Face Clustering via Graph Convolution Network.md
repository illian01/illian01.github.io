---
layout: post
title: Linkage Based Face Clustering via Graph Convolution Network
tag: [ComputerVision, FaceClustering]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" %}

<h3>CVPR, 2019</h3>

---

<h3>1. Introduction</h3>

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

<p align="center"><img src="/assets/Computer Vision/Linkage Based Face Clustering via Graph Convolution Network/figure1.png" width="80%"></p>

IJB-B-512 데이터셋에 대해서 k-NN을 적용하고, 다른 레이블끼리의 link들을 제거한 뒤, connected component들을 클러스터 결과로 사용한 결과이다.

**Pipeline.** 

- 모든 instance들을 pivot으로 사용한다. 각 pivot에 대해서 Instance Pivot Subgraph (IPS)를 구성해서 사용한다.
- IPS가 입력 데이터로, graph convolution network를 적용한다. 이 네트워크는 pivot과 이웃 사이의 link를 예측한다.
- 네트워크를 통해 weighted edge들을 얻었으면, 이들을 통합해 클러스터를 구축한다.

<h4>3.2. Construction of instance pivot subgraph</h4>

**Step 1: Node discovery.** pivot $$p$$에 대해서 주변의 node들을 IPS에 추가한다.
pivot을 기준으로 $$h$$-hop까지의 이웃들이 추가되는데, 각 $$i$$-hop에서는 $$k_{i}$$개의 최근접 이웃이 선택된다.
$$h=3, k_{1}=8, k_{2}=4, k_{3}=2$$라고 했을 때, 1-hop으로 pivot의 최근접이웃 8개, 각 1-hop node들의 최근접이웃 4개(총 32개), 마지막으로 각 2-htop node들의 최근접 이웃 2개(총 64)가 추가되어 IPS에 총합 104개의 node가 포함된다.

**Step 2: Node feature normalization.** pivot $$p$$와 주변의 node set $$V_{p}$$를 얻었다. 
pivot정보를 IPS에 encode해주기 위해, pivot 특징 $$x_{p}$$를 IPS의 모든 node에서 뺄셈한다.

$$\mathcal{F}_{p} = \left[ ...,x_{q}-x_{p},...\right]^{T}, \mbox{for all } q\in V_{p}$$

$$\mathcal{F}_{p}\in \mathbb{R}^{ \left| V_{p} \right| \times D} $$ 이다.

**Step 3: Adding edges among nodes.** node가 구성되었으니 edge만 추가하면 된다.


<h4>3.3. Graph convolutions on IPS</h4>


<h4>3.4. Link merging</h4>

---

<h3>4. Experiment</h3>