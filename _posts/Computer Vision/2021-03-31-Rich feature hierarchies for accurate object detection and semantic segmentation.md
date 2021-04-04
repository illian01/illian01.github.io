---
layout: post
title: Rich feature hierarchies for accurate object detection and semantic segmentation
tag: [ComputerVision, ObjectDetection]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" | append: page.title %}

<h3>1. Introduction</h3>

R-CNN이라는 이름은 Regions with CNN 으로 말그대로 Region proposal과 CNN을 합친 것이다.

---

<h3>2. Object detection with R-CNN</h3>

R-CNN은 3개의 모듈로 나뉜다.

1. Region proposal
2. Convolutional neural network
3. class specific linear SVMs


<h4>2.1. Module design</h4>

**Region proposals.** category-independent 하게 region proposal하는 여러 알고리즘이 있지만,
여기서는 selective search를 사용한다.

**Feature extraction.** Caffe로 구현된 AlexNet을 이용하여 227x227 입력 이미지를 4096 벡터로 추출한다.
고정된 입출력을 가지기 때문에 입력을 이에 다 맞춰줘야 하는데, 저자는 전체 픽셀들을 warping해서 227x227로 맞춰서 입력으로 준다.
$$p$$ 개의 픽셀만큼 bounding box를 확장시킨 후에 이를 warping한다고 한다($$p=16$$). 자세한 내용은 Appendix A에서 설명한다.

<p align="center"><img src="{{ img_path }}_figure1.png?raw=true" width="80%"></p>


<h4>2.2. Test-time detection</h4>

Test time에 selective search를 사용해 2000장의 region proposal을 구해서 사용한다.
각 입력은 CNN에 먹여지고, CNN의 출력이 SVM을 통과해 class에 대한 점수를 얻는다.
모든 점수들이 주어졌을 때, IoU(Intersection over Union)이 threshold보다 높은 region들에 대해서
Non-maximum supression을 시행한다. 즉, 겹치는 region들 중에서는 가장 score가 높은 것만 남게된다.

**Run-time analysis.** CNN을 사용하기 때문에 CNN의 장점을 가져서 run-time이 빨라진다.
CNN은 모든 클래스에 대해서 parameter들이 공유되고, 출력되는 특징 벡터가 저차원으로 줄어드는데, 이게 연산에 도움을 주기 때문이다.
요즘엔 CNN이 당연시되어서 대충 넘어가는 내용이다.

<h4>2.3. Training</h4>

**Supervised pre-training.** CNN을 ILSVRC2012 데이터셋으로 classification 학습을 먼저 시킨다.

**Domain-specific fine-tuning.** pretrain 된 CNN에서 classification layer를 떼어내고,
(N + 1)개로 classification 하는 층을 붙인다. N은 원하는 클래스의 개수, +1은 어디에도 속하지 않는 배경 클래스 이다.
region proposal 중 정답 데이터와 IoU가 0.5 이상인 경우에 positive(어떤 클래스), 아니면 negative(배경) 데이터로 학습한다.

lr 0.001 로 SGD를 사용하며, 각 배치는 32개의 positive, 96개의 background로 128개씩 사용한다.

**Object category classifiers.** fine-tune을 했으면 다시 classification layer를 떼어내고 SVM을 학습한다.
여기서는 데이터가 positive인지 negative인지 결정하는 IoU threshold를 0.3으로 쓴다. 경험적으로 찾은 값으로 보인다.
한번에 모든 값을 메모리에 올려서 학습을 할 수 없어서 standard hard negative mining method 라는 방법을 썻다는데, 모르니까 넘어간다.
Appendix B에서 왜 threshold를 다르게 썻는지, 또 왜 softmax를 사용한 분류 층을 버리고 SVM을 사용하는지 설명한다.

<h4>2.4. Results on PASCAL VOC 2010-12</h4>

<p align="center"><img src="{{ img_path }}_figure2.png?raw=true" width="100%"></p>

<h4>2.5. Results on ILSVRC2013 detection</h4>

<p align="center"><img src="{{ img_path }}_figure3.png?raw=true" width="100%"></p>

---

<h3>3. Visualization, ablation, and modes of error</h3>

<h4>3.1. Visualizing learned features</h4>

pass

<h4>3.2. Ablation studies</h4>

pass

<h4>3.3. Network architectures</h4>

다른 네트워크를 통한 실험을 보인다. T-Net이 AlexNet, O-Net이 VGGNet이다. VGGNet이 prediction에 7배 많은 시간이 걸렸다고 한다.

<p align="center"><img src="{{ img_path }}_figure4.png?raw=true" width="100%"></p>

<h4>3.4. Detection error analysis</h4>

DPM에 비해서 error가 어떤식으로 비교되는지 설명한다. 생략

<h4>3.5. Bounding-box regression</h4>

localization error를 줄이기 위해서 pool5애서 나온 특징들을 이용해 linear regression 을 학습한다.
자세한 내용은 Appendix C에 나온다. 표에 BB라고 써있는게 Bounding-box regression이 적용된 것이다.

<h4>3.6. Qualitative results</h4>

논문에 사진들을 보라는데, 양이 많아서 넘어간다.

---

<h3>4. The ILSVRC2013 detection dataset</h3>

pass

---

<h3>5. Semantic segmentation</h3>

pass

---

<h3>6. Conclusion</h3>

Object detection 분야에 CNN을 도입하면서 큰 성능 향상을 만들어냈다.

---

<h2>Appendix</h2>

<h3>A. Object proposal transformations</h3>

<p align="center"><img src="{{ img_path }}_figure5.png?raw=true" width="80%"></p>

(A) 원본
(B) 주변 context까지 가져와서 box를 채움
(C) context는 가져오지 않고 최대한 box를 채움(패딩)
(D) warping

각 이미지의 윗줄은 꽉 맞는 boinding box를 사용한 것이고, 아랫줄은 16 픽셀만큼 bounding box를 확장한 후 사용한 것이다.
16픽셀 확장 후 warping이 가장 성능이 좋았다고 한다.

<h3>B. Positive vs. negative examples and softmax</h3>



<h3>C. Bounding-box regression</h3>

localization 성능을 끌어올리기 위해 bounding-box regression 단계를 추가한다.
SVM을 통해서 클래스 점수가 구해지면, 그 클래스가 어떤 박스를 가지는지 간단하게 regression하는 것이다.
학습은 CNN의 특징 벡터를 사용한다.

입력은 $$\{ (P^{i}, G^{i}) \}_{i=1,...,N}$$ 이다. $$P^{i}=(P_{x}^{i}, P_{y}^{i}, P_{w}^{i}, P_{h}^{i})$$로 구성된다.
$$(x, y)$$는 proposal 의 중심좌표, $$(w, h)$$는 폭과 높이를 뜻한다. $$G$$는 ground-truth로 같은 구성을 가진다. 이후 표기에 $$i$$는 생략한다.

변환식은 다음과 같이 쓴다.

$$ 
\begin{align*} 
\hat{G}_{x} & = P_{w}d_{x}(P) + P_{x} \\
\hat{G}_{y} & = P_{h}d_{y}(P) + P_{y} \\
\hat{G}_{w} & = P_{w}exp(d_{w}(P)) \\
\hat{G}_{h} & = P_{h}exp(d_{h}(P)))
\end{align*}
$$

각 $$d_{*}(P)$$는 proposal P가 CNN의 pool5에서 나온 feature로 모델링된 선형 회귀 모델이다.
proposal P에서 나온 pool5의 feature를 $$\phi_{5} (P)$$라 하고, 학습 가능한 weight를 $$\mathbb{w}$$라 하면

$$ d_{*}(P) = \mathbb{w}_{*}^{T} \phi_{5}(P) $$

이 $$\mathbb{w}$$를 ridge regression으로 학습한다고 하면 다음과 같이 식을 쓸 수 있다.

$$ \mathbb{w}_{*} = \arg\!\min_{\hat{\mathbb{w}}_{*}} \sum_{i}^{N} (t_{*}^{i} - \hat{\mathbb{w}}_{*}^{T} \phi_{5}(P^{i}))^{2} + \lambda  \lVert \hat{\mathbb{w}}_{*} \rVert^{2} $$

$$t$$가 학습 목표가 되고 학습 쌍 $$(P, G)$$에 대해서 다음과 같이 정의된다.
이는 위의 G에 대한 식에서 $$d$$를 $$t$$로 두면 간단하게 만들어진다.

$$ 
\begin{align*} 
t_{x} & = (G_{x} - P_{x}) / P_{w} \\
t_{y} & = (G_{y} - P_{y}) / P_{h} \\
t_{w} & = \log(G_{w} / P_{w}) \\
t_{h} & = \log(G_{h} / P_{h})
\end{align*}
$$

이제 위의 regression 을 적용하기 위해 마지막 두가지 이슈만 정리하면 된다.

1. $$\lambda$$ 는 1000으로 사용한다.
2. $$(P, G)$$ 쌍을 너무 연관 없는 것까지 사용하면 학습에 악영향을 준다. 따라서 $$P$$와 $$G$$가 어느정도 가까운 경우에만 학습 쌍으로 사용한다.
 + 가깝다 == IoU가 0.6 이상이다.

---

후략