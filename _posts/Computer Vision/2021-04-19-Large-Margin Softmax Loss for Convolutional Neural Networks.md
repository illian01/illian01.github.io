---
layout: post
title: Large-Margin Softmax Loss for Convolutional Neural Networks
tag: [ComputerVision, FaceRecognition]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" %}

<h3>ICML, 2016</h3>

---

<h3>1. Introduction</h3>

intra class compactness와 inter class separability를 향상시키기 위해서 softmax를 일반화한 L-Softmax를 제안한다.

<p align="center"><img src="{{ img_path }}figure11.png?raw=true" width="80%"></p>
*그림은 MNIST에 대한 결과*

---

<h3>2. Related work and preliminaries</h3>

입력 특징 벡터 $$x_{i}$$와 출력 $$y_{i}$$가 있을 때 softmax는 다음과 같이 쓸 수 있다.

$$L = \frac{1}{N} \sum_{i} L_{i} = \frac{1}{N} \sum_{i} -\log (\frac{e^{fy_{i}}}{\Sigma_{j}e^{f_{j}}})$$

$$f_{j}$$는 클래스 점수 벡터 $$f$$의 $$j$$번째 값을, N은 데이터의 수를 의미한다.
그렇다면 $$f_{j}$$는 마지막 fully connected layer의 연산이라는 뜻이므로 weight를 $$W$$라 해서 $$W^{T}_{y_{i}}x_{i}$$로 쓸 수 있다(단순화를 위해 bias는 생략).
$$f_{i}$$가 앞의 설명에 따라 $$W$$와 $$x$$의 내적이므로, $$f_{j} = \lVert W_{j} \rVert \lVert x_{i} \rVert \cos(\theta_{j})$$라고도 쓸 수 있다.
$$\theta_{j}(0 \le \theta_{j} \le \pi)$$는 $$W_{j}$$와 $$x_{i}$$의 사이각이다. 위의 식을 다시쓰면 다음과 같이 된다.

$$L_{i} = - \log (\frac{e^{\lVert W_{y_{i}} \rVert \lVert x_{i} \rVert \cos(\theta_{y_{i}})}}{\Sigma_{j} e^{\lVert W_{j} \rVert \lVert x_{i} \rVert \cos(\theta_{j})}})$$

---

<h3>3. Large-margin softmax loss</h3>

<h4>3.1. Intuition</h4>

클래스가 2개만 있다고 했을 때, 기존의 softmax에서 $$x$$를 클래스 1번으로 분류하기 위해서는 $$W^{T}_{1}x > W^{T}_{2}x$$가 성립해야 한다.
즉, $$\lVert W_{1} \rVert \lVert x \rVert \cos(\theta_{1}) > \lVert W_{2} \rVert \lVert x \rVert \cos(\theta_{2})$$가 성립해야 한다.
여기서 좀 더 엄격한 조건을 걸어서 decision boundary에 마진을 형성하려는 것이 목적이다. 
여기에 하이퍼파라미터 $$m$$($$m$$은 양의 정수)을 첨가하면 $$\lVert W_{1} \rVert \lVert x \rVert \cos(m\theta_{1}) > \lVert W_{2} \rVert \lVert x \rVert \cos(\theta_{2})\ (0 \le \theta_{1} \le \frac{\pi}{m})$$를 목표로 학습할 수 있다.

$$\cos$$는 $$[0, \pi]$$에서 단조 감소하기 때문에 $$[0, \frac{\pi}{m}]$$에서 $$\lVert W_{1} \rVert \lVert x \rVert \cos(\theta_{1}) \ge \lVert W_{1} \rVert \lVert x \rVert \cos(m\theta_{1})$$이 성립한다.
따라서, $$\lVert W_{1} \rVert \lVert x \rVert \cos(m\theta_{1}) > \lVert W_{2} \rVert \lVert x \rVert \cos(\theta_{2})$$을 class 1을 위한 분류조건으로 하는 것이 더 엄격한 조건이 되는 것이다.

<h4>3.2. Definition</h4>

앞서의 설명에 따라서 L-Softmax loss는 다음과 같이 정의된다.

$$L_{i} = - \log \left( \frac{e^{\lVert W_{y_{i}} \rVert \lVert x_{i} \rVert \psi(\theta_{y_{i}})}}{e^{\lVert W_{y_{i}} \rVert \lVert x_{i} \rVert \psi(\theta_{y_{i}})} + \Sigma_{j \ne y_{i}} e^{\lVert w_{j} \rVert \lVert x_{i} \rVert \cos(\theta_{j})}} \right)$$

여기서 $$\psi$$는 다음과 같다.

$$
\psi(\theta) = 
\begin{cases}
\cos(m\theta), & 0 \le \theta \le \frac{\pi}{m} \\
\mathcal{D}(\theta), & \frac{\pi}{m} < \theta \le \pi
\end{cases}
$$

$$m$$은 클수록 학습 목표를 어렵게 만들고, decision boundary의 마진을 크게 만든다.
그리고 $$\mathcal{D}(\theta)$$는 $$\mathcal{D}(\frac{\pi}{m}) = \cos(\frac{\pi}{m})$$인 단조 감소 함수이다.
저자는 $$\mathcal{D}$$를 $$\cos$$의 감소부분만 반복시키도록 해서 다음과 같은 식을 구성한다.

$$\psi(\theta) = (-1)^{k} \cos(m\theta) - 2k, \ \ \theta \in \left[ \frac{k\pi}{m}, \frac{(k + 1)\pi}{m} \right]$$

$$k$$는 $$k \in [0, m - 1]$$인 정수이다.

<p align="center"><img src="{{ img_path }}figure1.png?raw=true" width="60%"></p>


<h4>3.3. Geometric interpretation</h4>

어떻게 angle margin을 극대화 하는 것인지 설명한다. 단순화를 위해 역시 2개의 클래스에 대해 분류하는 것으로 생각한다.

<p align="center"><img src="{{ img_path }}figure2.png?raw=true" width="60%"></p>

우선, $$\lVert W_{1} \rVert = \lVert W_{2} \rVert$$인 경우에 대해서 보자.
본래의 softmax에서 class 1로 분류하기 위해서는 $$\theta_{1} < \theta_{2}$$가 성립해야 한다. 
반면에 L-Softmax는 $$m\theta_{1} < \theta_{2}$$가 성립해야 하기 때문에 $$\theta_{1}$$이 훨씬 더 줄어들어야 class 1로 분류 할 수 있게 된다.
이 $$m$$은 각 클래스를 학습할 때 그 클래스에만 적용되어 들어가기 때문에 class 1들은 $$W_{1}$$에 붙게 되고, class 2들도 $$W_{2}$$에 붙게 되어서 dicision boundary 사이에 공간이 생기게 된다.
모든 학습 데이터가 완벽히 분류된다고 했을 때, 두 클래스의 angle margin은 $$\frac{m-1}{m+1}\theta_{1, 2}$$이다. $$\theta_{1, 2}$$는 $$W_{1}$$와 $$W_{2}$$의 사이각이다.

$$\lVert W_{1} \rVert \ne \lVert W_{2} \rVert$$인 경우에도 큰 차이는 없다.
$$\lVert W_{i} \rVert$$역시 분류에 있어서 중요한 값이기 때문에 decision 영역이 한쪽으로 좀 더 치우쳐져 있게 된다.
하지만 이 경우에서도 $$m$$값이 decision margin을 확보하기 때문에 여전히 intra class compactness는 줄이고, inter class separability는 늘리는 목적은 달성하고 있다.


<h4>3.4. Discussion</h4>

L-Softmax는 다음의 좋은 특성을 가지고 있다.

+ 분명한 기하학적 해석을 가지고 있다. $$m$$이 class간의 마진을 조정하며 클수록 이 마진도 커지고 학습난이도 역시 증가한다. softmax는 $$m=1$$인 경우로 볼 수 있다.
+ 어려운 학습은 overfitting을 피할 수 있게 한다.
+ 기존의 loss를 대체해서 쉽게 쓰일 수 있다.

---

<h3>4. Optimization</h3>

forward나 backward나 $$\theta$$를 지울 필요가 있다. 드무아브르 공식에 따라 $$\cos(m\theta_{y_{i}})$$는 다음으로 쓸 수 있다.

$$
\begin{align*}
\cos(m\theta_{y_{i}}) =& C^{0}_{m}\cos^{m}(\theta_{y_{i}}) - C^{2}_{m}\cos^{m-2}(\theta_{y_{i}})(1-\cos^{2}(\theta_{y_{i}})) + C^{4}_{m}\cos^{m-4}(\theta_{y_{i}})(1-\cos^{2}(\theta_{y_{i}}))^{2} \\
                      &+ \cdots +  (-1)^{n}C^{2n}_{m}\cos^{m-2n}(\theta_{y_{i}})(1-\cos^{2}(\theta_{y_{i}}))^{n} + \cdots
\end{align*}
$$

$$n$$은 $$2n \le m$$인 정수이다. $$\cos(\theta_{j})$$를 $$\frac{W^{T}_{j}x_{i}}{\lVert W_{j} \rVert \lVert x_{i} \rVert}$$로 바꾸면 $$W$$와 $$x$$만으로 식을 구성할 수 있다.
이걸 가지고 $$f_{y_{i}}$$를 다시 써보자.

$$
\begin{align*}
f_{y_{i}} =& \lVert W_{y_{i}} \rVert \lVert x_{i} \rVert \psi(\theta_{y_{i}}) \\
          =& (-1)^{k} \cdot \lVert W_{y_{i}} \rVert \lVert x_{i} \rVert \cos(m\theta_{i}) - 2k \cdot \lVert W_{y_{i}} \rVert \lVert x_{i} \rVert \\
          =& (-1)^{k} \cdot \lVert W_{y_{i}} \rVert \lVert x_{i} \rVert ( C^{0}_{m} (\frac{W^{T}_{y_{i}}x_{i}}{\lVert W_{y_{i}} \rVert \lVert x_{i} \rVert})^{m} 
          - C^{2}_{m} (\frac{W^{T}_{y_{i}}x_{i}}{\lVert W_{y_{i}} \rVert \lVert x_{i} \rVert})^{m-2}(1-(\frac{W^{T}_{y_{i}}x_{i}}{\lVert W_{y_{i}} \rVert \lVert x_{i} \rVert})^{2}) + \cdots)  \\
          &- 2k \cdot \lVert W_{y_{i}} \rVert \lVert x_{i} \rVert
\end{align*}
$$

필요한건 $$\frac{\partial L_{i}}{\partial x_{i}} = \Sigma_{j} \frac{\partial L_{i}}{\partial f_{j}} \frac{\partial f_{j}}{\partial x_{i}}$$ 와
$$\frac{\partial L_{i}}{\partial W_{y_{i}}} = \Sigma_{j} \frac{\partial L_{i}}{\partial f_{j}} \frac{\partial f_{j}}{\partial W_{y_{i}}}$$ 2개이다.
$$\frac{\partial f_{j}}{\partial x_{i}}$$와 $$\frac{\partial f_{j}}{\partial W_{y_{i}}}$$가 다음과 같이 계산된다고 한다. 직접 해보진 않았다. 수식 치기 힘들어서 사진으로 넣는다.

<p align="center"><img src="{{ img_path }}eq1.png?raw=true" width="50%"></p>

<p align="center"><img src="{{ img_path }}eq2.png?raw=true" width="50%"></p>

$$m=2$$로 예시를 들자면 $$f_{i}$$는 다음의 식이 된다.

$$f_{i} = (-1)^{k} \frac{2(W^{T}_{y_{i}}x_{i})^{2}}{\lVert W_{y_{i}} \rVert \lVert x_{i} \rVert} - (2k + (-1)^{k}) \lVert W_{y_{i}} \rVert \lVert x_{i} \rVert$$

$$\mbox{where, } k =
\begin{cases}
1, & \frac{(W^{T}_{y_{i}}x_{i})}{\lVert W_{y_{i}} \rVert \lVert x_{i} \rVert} \le \cos(\frac{\pi}{2})\\
0, & \frac{(W^{T}_{y_{i}}x_{i})}{\lVert W_{y_{i}} \rVert \lVert x_{i} \rVert} > \cos(\frac{\pi}{2})
\end{cases}
$$

$$\frac{\partial f_{j}}{\partial x_{i}}$$와 $$\frac{\partial f_{j}}{\partial W_{y_{i}}}$$는 다음과 같이 계산된다.

<p align="center"><img src="{{ img_path }}eq3.png?raw=true" width="50%"></p>

---

<h3>5. Experiments and results</h3>

<h4>5.1. Experimental settings</h4>

MNIST, CIFAR10, CIFAR100, LFW 데이터셋에 대해서 평가한다. 사용되는 CNN은 다음과 같다.

<p align="center"><img src="{{ img_path }}figure3.png?raw=true" width="100%"></p>

**General settings.** SGD, PReLU, batch_size=256, weight_decay=0.0005, momentum=0.9, weight_init=he_normal, batch_normalization(no dropout), mean substraction.

데이터의 subject가 너무 많은경우 식을 조금 수정해서 사용한다.

$$f_{y_{i}} = \frac{\lambda \lVert W_{y_{i}} \rVert \lVert x_{i} \rVert \cos(\theta_{y_{i}}) + \lVert W_{y_{i}} \rVert \lVert x_{i} \rVert \psi(\theta_{y_{i}})}{1+\lambda}$$

$$\lambda$$를 학습초기에 크게 했다가(그냥 softmax과 비슷해짐) 학습 진행에 따라 0까지 낮추면서 학습한다.

**MNIST, CIFAR10, CIFAR100.** lr=0.1로 시작해서 10k, 12k, 15k iteration에서 10으로 나눔, 18k iteration까지 학습

**Face Verification.** lr=0.1, 0.01, 0.001로 loss가 크게 변동이 없으면 낮춤.

**Testing.** MNIST, CIFAR10, CIFAR100은 softmax와 비교. LFW는 cosine distance와 nearest neighbor rule 사용.

<h4>5.2. Visual classification</h4>

**MNIST.**

<p align="center"><img src="{{ img_path }}figure4.png?raw=true" width="50%"></p>

**CIFAR10.**

<p align="center"><img src="{{ img_path }}figure5.png?raw=true" width="50%"></p>

**CIFAR100.**

<p align="center"><img src="{{ img_path }}figure6.png?raw=true" width="50%"></p>

**Confusion matrix visualization.**

<p align="center"><img src="{{ img_path }}figure7.png?raw=true" width="80%"></p>

**Error rate vs. Iteration.**

CIFAR100에 대한 softmax와 L-Softmax의 차이: 좌측이 train error, 우측이 test error

<p align="center"><img src="{{ img_path }}figure8.png?raw=true" width="80%"></p>

상대적으로 큰 CNN과 작은 CNN에서의 L-Softmax 성능 차이(overfitting에 대한 실험): 좌측이 train error, 우측이 test error

<p align="center"><img src="{{ img_path }}figure9.png?raw=true" width="80%"></p>


<h4>5.3. Face verification</h4>

학습은 CASIA-WebFace 데이터셋(10000명에 대해서 490k의 이미지 포함)으로 진행. 
preprocessing으로 IntraFace를 사용해서 얼굴을 정렬 후 crop해서 사용한다. 추출된 특징을 PCA로 압축해 사용한다.

<p align="center"><img src="{{ img_path }}figure10.png?raw=true" width="50%"></p>

---

<h3>6. Concluding remarks</h3>

pass