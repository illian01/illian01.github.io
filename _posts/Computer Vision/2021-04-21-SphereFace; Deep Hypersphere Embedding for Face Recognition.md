---
layout: post
title: SphereFace; Deep Hypersphere Embedding for Face Recognition
tag: [ComputerVision, FaceRecognition]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" %}

<h3>CVPR, 2017</h3>

---

<h3>1. Introduction</h3>

L-Softmax에서 조금 더 변형한 것을 open-set 데이터에 대해서 적용하고, 왜 open-set에 잘 작동하는지 보인다.

<p align="center"><img src="{{ img_path }}figure1.png?raw=true" width="70%"></p>

closed-set과 달리 open-set은 metric과 관련이 된다.
학습 데이터에 대해서 intra class 거리를 줄이고 inter class 거리를 늘려서 구분을 지어줘야 본 적 없는 클래스에 대해서도 어떤 클래스에도 해당하지 않는다고 말할 수 있기 때문이다.
기존의 Euclidean margin 대신 Angular margin을 사용한다. 주요 내용은 다음과 같다.

1. A-Softmax의 소개
2. lower bound $$m$$에 대한 설명
3. SphereFace의 학습과 결과

<p align="center"><img src="{{ img_path }}figure2.png?raw=true" width="100%"></p>


---

<h3>2. Related work</h3>

pass

---

<h3>3. Deep hypersphere embedding</h3>

<h4>3.1. Revisiting the softmax loss</h4>

입력 특징 $$x_{i}$$와 출력 $$y_{i}$$에 대해서 softmax는 다음과 같다.

$$L = \frac{1}{N} \sum_{i} L_{i} = \frac{1}{N} \sum_{i} -\log (\frac{e^{f_{y_{i}}}}{\Sigma_{j} e^{f_{j}}})$$

$$f_{j}$$는 클래스 점수 벡터의 $$j$$번째 값이다. 즉, $$f$$는 $$f_{j} = W^{T}_{j}x_{i} + b_{j}$$로 쓸 수 있다. 
또, $$W^{T}_{i}x = \lVert W^{T}_{i} \rVert \lVert x \rVert \cos(\theta_{i})$$이므로 $$f_{j} = \lVert W^{T}_{j} \rVert \lVert x_{i} \rVert \cos(\theta_{j, i}) + b_{j}$$ 이다.
$$\theta_{j, i}(0 \le \theta_{j, i} \le \pi)$$는 $$W_{j}$$와 $$x_{i}$$의 사이각이다. 따라서 $$i$$번째 입력의 loss는

$$
\begin{align*}
L_{i} &= -\log ( \frac{e^{W^{T}_{y_{j}}x_{i}+b_{y_{i}}}}{\Sigma_{j} e^{W^{T}_{j}x_{j}+b_{j}}} ) \\
      &= -\log( \frac{e^{\lVert W_{y_{i}} \rVert \lVert x_{i} \rVert \cos(\theta_{y_{i}, i})+b_{y_{i}}}}{\Sigma_{j} e^{\lVert W_{j} \rVert \lVert x_{i} \rVert \cos(\theta_{j, i})+b_{j}}} )
\end{align*}
$$

가 된다. 이걸 $$\forall_{j}$$에 대해서 $$\lVert W_{j} \rVert = 1, b_{j} = 0$$로 normalize한다. 이러면 $$\lVert x \rVert$$와 $$\cos(\theta_{j, i})$$만으로 분류할 수 있게 된다.

$$L_{\mbox{modified}} = \frac{1}{N} \sum_{i} -\log( \frac{e^{\lVert x_{i} \rVert \cos(\theta_{y_{i}, i})}}{\Sigma_{j} e^{\lVert x_{i} \rVert \cos(\theta_{j, i}}} )$$


<h4>3.2. Introducing angular margin to softmax loss</h4>

기본적인 아이디어는 decision boundary를 조정해서 angular margin을 만들자 라는 것이다. $$L_{\mbox{modified}}$$에서 이진 분류를 위해 $$W_{1}, W_{2}$$가 있다고 하자.
$$x$$가 주어지면 class 1로 분류하기 위해서는 $$\cos(\theta_{1}) > \cos(\theta_{2})$$가 성립해야 한다.
하지만 여기 $$m(m \ge 2)$$이 주어져서 $$\cos(m\theta_{1}) > \cos(\theta_{2})$$를 성립시켜야 한다면 $$\theta_{1}$$이 더 작아저야 한다.
이러한 원리로 $$m$$이 decision boundary에 margin을 형성하게 되고 학습 데이터가 모듀 분류되었을 때 angular margin은 $$\frac{m-1}{m+1}\theta^{1}_{2}$$이다.
$$\theta^{1}_{2}$$는 $$W_{1}$$과 $$W_{2}$$의 사이각이다. 그럼

$$L_{ang} = \frac{1}{N} \sum_{i} -\log (\frac{e^{\lVert x_{i} \rVert \cos(m\theta_{y_{i}, i})}}{e^{\lVert x_{i} \rVert\cos(m\theta_{y_{i}, i})} + \Sigma_{j \ne y_{i}} e^{\lVert x_{u} \rVert \cos(\theta_{j, i})}})$$

이 식을 얻을 수 있다. $$\theta_{y_{i}, j}$$는 $$[0, \frac{\pi}{m}]$$의 범위를 가진다. 
근데 이러면 원래 $$\theta$$는 $$[0, \pi]$$의 범위를 가져서 문제가 생기기 때문에 단조 감소 함수 $$\psi(\theta_{y_{i}, j})$$를 정의하는데, $$[0, \frac{\pi}{m}]$$에서는 $$\cos(\theta_{y_{i}, j})$$와 같도록 한다.

$$L_{ang} = \frac{1}{N} \sum_{i} -\log (\frac{e^{\lVert x_{i} \rVert \psi(m\theta_{y_{i}, i})}}{e^{\lVert x_{i} \rVert\psi(m\theta_{y_{i}, i})} + \Sigma_{j \ne y_{i}} e^{\lVert x_{u} \rVert \cos(\theta_{j, i})}})$$

$$\psi(\theta_{y_{i},j}) = (-1)^{k} \cos(m\theta_{y_{i}, j}) - 2k,\ \theta_{y_{i}, j} \in [\frac{k\pi}{m}, \frac{(k+1)\pi}{m}]\ \mbox{and}\ k \in [0, m-1]$$


<h4>3.3. Hypersphere interpretation of A-Softmax loss</h4>

<p align="center"><img src="{{ img_path }}figure3.png?raw=true" width="60%"></p>

손실 함수들의 기하학적 특성을 2d와 3d의 경우로 보인다. A-Softmax 부분을 보면 $$\theta$$가 $$\omega$$에 해당된다.
$$x$$는 $$\omega$$가 짧은쪽으로 분류되는데, 여기에 $$m$$을 추가한다면 $$\omega$$가 더 줄어들어야 해서 decision의 경계가 더 작아지게 된다.
modified softmax나 A-Softmax가 구형을 하는 것은 $$\lVert W \rVert$$를 1로 전부 통일했기 때문이다.

<h4>3.4. Properties of A-Softmax loss</h4>

적당한 $$m$$에 대한 수학적 접근이다. 넘어간다.

pass

<h4>3.5. Discussion</h4>

**Why angular margin.** softmax는 이미 본질적으로 각도가 포함되어있다. 
따라서 Euclidean margin을 부여하는 것보다 angular margin을 부여하는 것이 더 자연스럽다.

**Comparison with existing losses.** 기존의 contrastive loss나 triplet loss, center loss처럼 feature 공간에서 바로 거리 계산을 하는 방법이 있으나
이 것들은 데이터가 늘어갈수록 학습쌍을 만들어주는 것이 어려워지게 된다.

---

<h3>4. Experiments (more in Appendix)</h3>

<h4>4.1. Experimental settings</h4>

**Preprocessing.** 얼굴 검출에는 MTCNN을 사용한다. 주어진 얼굴 이미지는 [-1, 1] 값으로 바운드 해서 사용한다.

**CNNs setup.** SphereFace의 기본적인 구조는 다음과 같다.

<p align="center"><img src="{{ img_path }}figure4.png?raw=true" width="50%"></p>

비교하는 method들은 같은 CNN을 기준으로 한다. Section 3.4에 따라서 $$m=4$$를 사용한다.

<p align="center"><img src="{{ img_path }}figure5.png?raw=true" width="100%"></p>

S2는 stride가 2, FC1은 fully connected를 의미한다. residual 도 썼다는데, 정확히 어디어디 들어가는지는 모르겠다.

batch_size=128, lr=0.1 -> 16K, 24K iteration에서 10으로 나눔, 28K iteration까지 학습.

**Training data.** CASIA-WebFace 데이터셋을 사용한다. 10,575 명에 대한 494,414개의 얼굴 영상이 담겨있다. 
0.49M개의 학습 데이터는 DeepFace나 VGGFace, FaceNet의 것과 비교해서 적은 수이다.

**Testing.** 최종 특징은 원본의 특징과 수평으로 뒤집은 것의 특징을 연결한 것이다. 두 특징의 metric은 cosine distance로 계산한다. 
분류는 nearest neighbor classifier와 thresholding이 사용된다.


<h4>4.2. Exploratory experiments</h4>

**Effect of m.** $$m$$이 클수록 margin을 늘린다는 것을 보이는 간단한 예제이다.
CASIA-WebFace의 6명의 이미지로 실험을 한다. 출력 특징을 3짜리로 수정한 뒤, 샘플을 시각화한다.

<p align="center"><img src="{{ img_path }}figure6.png?raw=true" width="100%"></p>

LFW와 YTF에 대해서 정확도를 비교한다.

<p align="center"><img src="{{ img_path }}figure7.png?raw=true" width="50%"></p>

**Effect of CNN architectures.** CNN의 스펙은 위에 적힌 표를 따른다. 좌측이 LFW, 우측이 YTF이다.

<p align="center"><img src="{{ img_path }}figure8.png?raw=true" width="70%"></p>


<h4>4.3. Experiments on FLW and YTF</h4>

공정함을 위해 모든 method는 위 CNN 중 64-layer짜리를 사용해 구현된다.

<p align="center"><img src="{{ img_path }}figure9.png?raw=true" width="50%"></p>


<h4>4.4. Experiments on MegaFace challenge</h4>

<p align="center"><img src="{{ img_path }}figure10.png?raw=true" width="50%"></p>

protocol 은 train set 의 크기를 말한다. Rank1 Acc.은 Identification task의 Rank1 정확도를 말한다. 
Ver은 verification task가 TAR(True Accept Rate) for $$10^{-6}$$ FAR(False Accept Rate)일 때 라는데 FAR이 낮을수록 더 엄격한 조건이 된다는 것 말고는 모르겠다.

<p align="center"><img src="{{ img_path }}figure11.png?raw=true" width="100%"></p>

ROC curve는 알겠는데 CMC curve는 처음들어본다.

---

<h3>5. Concluding remarks</h3>

pass

---

<h3>Appendix</h3>


<h4>A. The intuition of removing the last ReLU</h4>

보통 Conv layer들이 ReLU들을 통과하며 FC까지 가는데, FC에 연결되는 마지막 Conv layer까지 ReLU를 쓰면 FC가 활용할 수 있는 공간(각도)이 줄어들게 된다.
이를 해결하기 위해 마지막 Conv layer는 ReLU를 지워서 사용한다.

<p align="center"><img src="{{ img_path }}figure12.png?raw=true" width="100%"></p>

<h4>B. Normalizing the weights could reduce the prior caused by the training data imbalance</h4>

<p align="center"><img src="{{ img_path }}figure13.png?raw=true" width="100%"></p>

그래프를 보면 샘플 수가 많인 클래스 일수록 weight의 norm이 크게 학습되는 경향을 볼 수 있다. 
이 것이 학습 데이터에 숨어져 있는 prior를 학습했다고 볼 수 있는데, face verification과 같은 경우에서는 prior가 validation에는 좋지 않은 영향을 줄 수 있다.
이 prior를 죽이기 위해서 마지막 FC의 weight는 normalize 해서 사용한다.

<h4>C. Empirical experiment of zeroing out the biases</h4>

A-Softmax가 각도에 대해서 해석되기 때문에 bias가 살아있으면 해석이 어려워진다. 
아래는 CASIA-WebFace를 학습한 뒤 bias를 조사한 것인데, 대부분이 0으로 나온 것으로보아 face verification에는 그다지 유용한 값이 아님을 알 수 있다.

<p align="center"><img src="{{ img_path }}figure14.png?raw=true" width="60%"></p>

MNIST를 시각화 한 것에서도 bias의 유무에 크게 변화는 보이지 않는다.

<p align="center"><img src="{{ img_path }}figure15.png?raw=true" width="100%"></p>


<h4>D. 2D visualization of A-Softmax loss on MNIST</h4>

A-Softmax의 $$m$$값에 따른 차이를 보인다.

<p align="center"><img src="{{ img_path }}figure16.png?raw=true" width="100%"></p>

<h4>E. Angular Fisher score for evaluating the feature discriminativeness and ablation study on our proposed modifications</h4>

Angular Fisher score(AFS)를 제안한다.

$$AFS = \frac{S_{w}}{S_{b}}$$

whithin-class scatter value $$S_{w}=\Sigma_{i} \Sigma_{x_{j}\in X_{i}}(1- \cos \left\langle x_{j}, m_{j} \right\rangle)$$,
between-class scatter value $$S_{b}=\Sigma_{i} n_{i} (1- \cos \left\langle m_{i}, m \right\rangle)$$로 정의된다. 
$$X_{i}$$는 $$i$$번째 클래스 샘플, $$m_{i}$$는 클래스 $$i$$의 특징들에 대한 mean vector, $$m$$은 전체 데이터셋의 mean vector, $$n_{i}$$는 클래스 $$i$$의 샘플 번호이다.
일반적으로 점수가 낮을수록 더 구분이 잘 된다는 뜻으로 통한다.

앞서 설명한 방법들을 적용했을 때 결과가 어떻게 변하는지 LFW 데이터셋에 실험한다. 학습은 4층짜리 CNN과 CASIA 데이터셋을 사용한다.


<p align="center"><img src="{{ img_path }}figure17.png?raw=true" width="100%"></p>

<h4>F. Experiments on MegaFace with different convolutional layers</h4>

CNN의 깊이에 따른 변화. $$m=4$$를 쓴다.

<p align="center"><img src="{{ img_path }}figure18.png?raw=true" width="80%"></p>

<h4>G. The annealing optimization strategy for A-Softmax loss</h4>

L-Softmax에서와 같은 내용이다. 처음부터 어렵게 학습하면 학습이 잘 안된다. 
$$\lambda$$값을 두고 이 값을 처음에 크게 설정했다가 0으로 줄여가면서 학습하면 softmax에서 A-Softmax로 전환하면서 학습하는 효과를 얻을 수 있다.

$$f_{y_{i}} = \frac{\lambda \lVert x_{i} \rVert \cos(\theta_{y_{i}}) + \lVert x_{i} \rVert \psi(\theta_{y_{i}})}{1 + \lambda}$$

<h4>H. Details of the 3-patch ensemble strategy in MegaFece challenge</h4>

MegaFace challenge에 3-patch ensemble이 있는데, 이건 3개의 patch를 특징으로 뽑아서 concat한 것을 사용한 것이다.

<p align="center"><img src="{{ img_path }}figure19.png?raw=true" width="70%"></p>