---
layout: post
title: Deep Residual Learning for Image Recognition
tag: [ComputerVision, Unclassified]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" %}

<h3>1. Introduction</h3>

2015년도 ILSVRC 1위의 주인공 ResNet 논문이다. 
네트워크에서 깊이가 깊다는 것은 다양한 level의 feature를 가진다는 뜻이다. 
그러면 Layer를 쌓기만 하면 학습이 잘 되는 것인가? 라는 생각이 들지만 문제점이 존재한다. 

우선 Vanishing/Exploding Gradients 문제가 있는데, 이는 Normalized initialization이나 
Intermediate normalization layer와 같은 방법을 통해서 개선되어 왔다.

두번째 문제가 Degradation problem인데, 이는 더 깊은 모델이 train/test 모두에서 더 높은 error를 보이는 경우이다.
즉, 학습이 더 안되는 것이다.

<p align="center"><img src="{{ img_path }}figure1.png?raw=true" width="80%"></p>
*CIFAR-10에 대한 다른 두 모델의 error rate*

이 문제는 Overfitting과는 다르다. Overfitting이라면 train의 error가 줄고 test의 error가 증가하는 모습을 보여야하는데, 둘 다 수렴하는 그래프를 그리고 있다.

이 Degradation problem을 해결하기 위한 것이 Deep residual learning framework이다. 


---

<h3>2. Related Work</h3>

pass

---

<h3>3. Deep Residual Learning</h3>

<h4>3.1 Residual Learning</h4>

어떤 쌓여있는 레이어들이 학습하려는 매핑을 $$\mathcal{H}(x)$$라 하자. 여기서 $$x$$는 첫 번째 층의 입력이다.
residual learning은 여러개의 nonlinear layer가 어떠한 complicated function에 수렴할 수 있다면, residual function에도 수렴할 수 있다는 가정에서 시작한다. 
즉, $$\mathcal{H}(x) - x$$를 학습할 수 있다는 뜻으로(input과 output의 dimention이 같다고 가정했을 때), $$\mathcal{F}(x) := \mathcal{H}(x) - x$$로 쓴다. 
$$\mathcal{F}(x)$$를 구했다면 원래 목적인 $$\mathcal{H}(x)$$는 $$\mathcal{F}(x) + x$$로 얻을 수 있다. 대신 학습의 난이도는 다를 것이다.

identity가 optimal일 때, 복잡한 구조가 수렴에 어려움을 겪는 것이 아닌가 하는 것에서 착안했다고 하는데, 이정도에서 넘어가자.

<h4>3.2 Identity Mapping by Shortcuts</h4>

<p align="center"><img src="{{ img_path }}figure2.png?raw=true" width="50%"></p>

위의 그림을 building block으로 쌓아서 모델을 구축한다. 

$$ y = \mathcal{F}(x, \{ W_{i} \} ) + x $$

$$\mathcal{F} (x, \{ W_{i} \})$$가 나머지를 구하는 표현이 되겠다.
$$x$$를 더하는 것은 각 원소간 덧셈이다. 그런데 네트워크를 지나다 보면 dimension이 바뀌는 경우가 있다.
이럴 땐 linear projection $$W_{s}$$를 가져와서 $$x$$에 적용해 같은 차원을 가지도록 맞춰준다.

$$ y = \mathcal{F} (x, \{ W_{i} \}) + W_{s}x $$

본문에서는 $$\mathcal{F}$$를 2~3개의 층으로 묶어서 표현했으나, 몇 개 단위로 하던 상관은 없다.
단, 1개인 경우에는 shortcut의 의미가 사라지게된다.

표기는 fully-connected layers로 되어있으나, convolutional layers에 대해서도 똑같이 적용된다.

<h4>3.3 Network Architectures</h4>

plain과 residual 모델로 나눠서 실험한다.

**Plain Network.** VGG net을 두 가지 규칙을 따라서 변형해서 사용한다.
(i) 각 레이어는 같은 필터 수를 가진다. (ii) feature map 의 크기가 반이 되면, 필터 수는 2배가 된다(레이어의 시간 복잡도 유지).
끝에는 global average pooling과 1000개짜리 fully-connected layer가 붙는다.

**Residual Network.** plain network에서 short connection들이 붙는다.
입출력의 dimension이 같다면 identity가 붙으면 되고, dimension이 증가하는 경우에는 2가지 옵션이 있다.
(A) shortcut은 identity를 가지되, 나머지 차원에는 zero padding을 한다. 
(B) 1x1 convolution을 통해서 차원을 늘린다.

<p align="center"><img src="{{ img_path }}figure3.png?raw=true" width="50%"></p>
*Left: VGG-19(19.6 billion FLOPs). Middle: 34 layers plain network(3.6 billion FLOPs). Right: 34 layers residual network(3.6 billion FLOPs)*

<h4>3.4 Implementation</h4>

Train
1. Augmentation
 - 이미지의 짧은 쪽이 [256,480]중 랜덤한 숫자가 되도록 scale augmentation
 - random horizontal flip, per pixel mean subtracted.
 - 위 2개 적용 후에 224x224 random crop
 - standard color augmentation
2. convolution후, activation 전에 batch normalization 적용
3. He initilization
4. SGD with a mini-batch size of 256.
5. lr=0.1, is divided by 10 when the error plateaus.
6. weight decay=0.0001, momentum=0.9
7. $$60*10^4$$ iteration, no dropout

Test
1. standard 10-crop testing
2. fully-convolutional form을 가져온다.
3. 이미지의 짧은 쪽이 {224, 256, 384, 480, 640}로 rescale한 것들을 score average

---

<h3>4. Experiments</h3>

<h4>4.1 ImageNet Classification</h4>

ImageNet 2012 classification dataset을 사용:
class 1000, 1.28 million train image, 50k validation image, 100k test image.

<p align="center"><img src="{{ img_path }}figure4.png?raw=true" width="90%"></p>

**Plain Networks.** 위에 소개된 34층짜리와 비슷한 방식으로 설계된 18층 짜리를 비교한다.
34층 짜리 모델이 학습 전반에 걸쳐서 training/validaiton 모두 error가 높은 것을 볼 수 있다(degradation).
그래프에서 굵은 선이 validation error, 얇은 선이 training error이다.

<p align="center"><img src="{{ img_path }}figure5.png?raw=true" width="60%"></p>

**Residual Networks.** plain의 18, 34층과 같은 형태에 shortcut connection만 추가한다.
여기서 차원이 바뀔 때 shortcut은 identity mapping에 zero padding을 사용한다.
즉, 추가되는 파라미터는 없다. 34층 모델이 18층 모델모다 학습 전반에서 에러가 낮은 것을 볼 수 있다.

<p align="center"><img src="{{ img_path }}figure6.png?raw=true" width="60%"></p>

residual network를 사용했을 때, 깊은 모델이 더 낮은 training error 와 좋은 generalization을 보여준다.
이는 degradation problem이 잘 해결되었음을 뜻한다.
또한 18층짜리 plain 과 resnet을 비교해보면 resnet이 더 빠르게 수렴하는 것을 알 수 있다. 

<p align="center"><img src="{{ img_path }}figure7.png?raw=true" width="40%"></p>

**Identity vs. Projection Shortcuts.** 위에 입출력의 차원이 달라질 때 2가지의 옵션이 있다고 했는데,
이제 모델 전반에 걸쳐서 shortcut을 적용할 것을 생각해보면 3가지를 생각해 보면 3가지가 있다.
(A) 모든 shortcut에 zero padding을 적용한다. 파라미터 증가는 없다.
(B) 1x1 convolution을 이용한 projection을 차원이 변화할 때만 적용한다. 그 외는 identity를 사용한다.
(C) 모든 shortcut에 projection을 사용한다.

(B)가 (A)보다, (C)가 (B)보다 나은 성능을 보여준다.
하지만 이 3가지 차이가 degradation을 해결하는데 결정적인 역할을 하는 것은 아니다.
이후의 설명에 C는 memory/time을 아끼기위해 사용하지 않는다.

<p align="center"><img src="{{ img_path }}figure8.png?raw=true" width="50%"></p>

**Deeper Bottleneck Architectures.** 더 깊은 모델을 구축하되, training time을 많이 늘리지 않기 위해
building block을 bottleneck design으로 수정한다.
3x3 필터를 가진 2개의 레이어에서 1x1, 3x3, 1x1 3개의 레이어를 가진 것으로 변경한다.
각 1x1 레이어는 차원을 늘리거나 줄이는 하는 역할을 가진다. 3x3 레이어는 상대적으로 작은 데이터 양만 다루게 되는 것이다.

<p align="center"><img src="{{ img_path }}figure9.png?raw=true" width="70%"></p>

50-layer ResNet: 34층 resnet의 block들을 2개 레이어에서 3개 레이어로 다 바꾸면 50층이 된다.
shortcut은 (B) 옵션을 사용한다. 3.8 billion FLOPs를 가진다.

101-layer and 152-layer ResNets: block을 더 쌓아서 101층, 152층을 구성한다.
152층 ResNet은 11.3billion FLOPs를 가진다. VGG-19가 19.6 billion FLOPs를 가지는 것에 비해 여전히 작다.

50층, 101층, 152층 으로 점점 더 깊게 모델을 구축할 수록 에러가 줄어드는 모습을 볼 수 있다.

<p align="center"><img src="{{ img_path }}figure10.png?raw=true" width="70%"></p>

**Comparisons with State-of-the-art Methods.** 3.57% top-5 error로 ILSVRC 2015 1등했다.

<p align="center"><img src="{{ img_path }}figure11.png?raw=true" width="70%"></p>

<h4>4.2 CIFAR-10 and Analysis</h4>

pass

<h4>4.3 Object Detection on PASCAL and MS COCO</h4>

pass