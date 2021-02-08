---
layout: post
title: Very Deep Convolutional Networks For Large-Scale Image Recognition(VGGNet)
tag: [ComputerVision]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" | append: page.title %}

<h3>1. Introduction</h3>

VGGNet은 ILSVRC에서 1등을 한 적이 있는 모델이다. 
이전의 모델들 보다 훨씬 깊은 네트워크를 구축한게 특징인데, 이 논문도 깊이에 따른 성능 영향에 대해서 설명한다.

---

<h3>2. ConvNet Configurations</h3>

깊이에 따른 성능 차이를 비교할 것이기에 깊이를 제외한 설정을 다 같게 해야 할 것이다. 

<h4>2.1 Architecture</h4>

입력은 224x224 RGB 이미지를 받는다. 전처리는 훈련 셋 이미지들의 각 픽셀 값에서 RGB 값의 평균을 빼는 것만 진행한다. 각 conv layer의 filter는 3x3을 사용한다(1x1을 쓰는 부분도 있다). stride는 1로 고정이고 padding을 사용해서 conv로 인해 이미지가 줄어들지 않도록 한다. conv layer들의 뒤에 도중도중 등장하는 5개의 Max-pooling layer가 이미지를 줄이는 역할을 한다. Max-pooing은 2x2 window에 stride는 2를 가진다.

conv layers들이 끝나면 3개의 Fully-Connected(FC) layer가 따라온다. 앞의 2개는 4096 출력을 가지고, 마지막 층이 1000개의 출력을 가진다. ILSVRC에서의 class 개수가 1000개라서 그렇다. 이 FC layer들은 아래 제시되는 모든 모델에 똑같이 적용된다.

모든 hidden layer들은 ReLU를 사용하고 마지막에는 softmax가 적용된다.

<h4>2.2 Configurations</h4>

5개의 모델이 제시되는데 A~E로 이름을 붙인다. A가 11개의 layer로 제일 작고, E가 19개의 layer로 제일 많다. 

<h4>2.3 Discussion</h4>

VGGNet은 사실 이전에 ILSVRC에서 뛰어난 성과를 보였던 모델과는 다른 방식으로 짜여져 있다. 이전에는 filter size를 키워서 receptive field를 늘렸다면 VGGNet은 작은 filter로 대체하는 것을 제안한다. 2개의 layer가 3x3 filter로 쌓여있다면 5x5와 같은 receptive field를 가진다. 3개가 쌓여있다면 7x7 filter와 같은 영역을 볼 수 있다.

<p align="center"><img src="{{ img_path }}_figure1.png?raw=true" width="80%"></p>


그래서 3x3 filter를 쌓는 것이 큰 filter를 쓰는 것보다 어떤 장점이 있는 것일까? 

우선, 7x7 필터를 가진 conv layer 1개를 3x3 필터를 가진 conv layer 3개로 바꾼다면 activation function의 적용 회수가 달라진다. 이는 비선형성의 증가로 이어지고 같은 영역에서 더 많은 분류를 해낸다는 것을 뜻한다. 두 번째로는 파라미터 개수의 차이이다. 7x7 filter는 49개의 파라미터를 가지고, 3x3 filter 3개는 27개의 파라미터를 가진다(channel이 여기에 붙으면 차이는 더 커진다).

<p align="center"><img src="{{ img_path }}_figure2.png?raw=true" width="80%"></p>

---

<h3>3. Classification Framework</h3>

<h4>3.1 Training</h4>

multinomial logistic regression, batchsize=256, SGD(momentum=0.9, weight_decay=5e-4, lr=0.01(val acc가 향상되지 않으면 1/10)), dropout(3개의 FC 중 앞의 2개)=0.5

처음에 모델 A를 학습시키고 더 깊은 모델을 학습시킬 때 모델 A의 학습된 weight를 가져와 학습을 시작하는 방식을 사용한다. weight 초기화는 평균 0, 분산 0.01을 가지는 정규 분포에서 랜덤으로 가져오고, bias는 0으로 초기화된다.

-> 그런데 사실 알고보니 Glorot&Bengio(2010)의 weight 초기화 방식과 결과의 차이가 없다고 한다.

224x224 입력 이미지에서 augment는 random crop(one crop per image per SGD iter), random horizontal flipping, random RGB color shift가 train에 사용된다. 

**Training image size.** crop을 적용하는데 이미지를 어느정도 큰 이미지로 resize 후에 224x224를 crop하는 방식을 사용한다. 이 resize하는 크기를 S라고 한다(SxS img). 이 S를 고정해서 사용하는 방식과 랜덤으로 사용하는 방식 2가지를 쓴다. 고정하는 방식에서는 S=256, 384 두 값을 사용하고 랜덤값은 [256, 512] 범위의 값을 사용한다.

<h4>3.2 Testing</h4>

테스트 에서는 test scale Q가 있다. 이미지를 QxQ로 resize해서 사용하는데 S와 같을 필요는 없다. train때와는 다른 사이즈를 사용하기 때문에 FC layer들을 수정할 필요가 있다. Overfeat(Sermanet et al. 2014)라는 알고리즘을 통해서 3개의 FC layer 중 첫 번째 는 7x7 conv layer로 나머지는 1x1 conv layer로 바꾼다. 그렇다면 마지막은 class 수에 상응하는 channel 수와 spatial resolution이 남는다. pooling을 한다면 고정된 사이즈의 벡터를 얻을 수 있다.

<h4>3.3 Implementation Details</h4>

pass

---

<h3>4. Classification Experiments</h3>

**Dataset.** ILSVRC-2012 dataset으로 학습해서 top-1, top-5 error를 비교한다.

<h4>4.1 Single Scale Evaluation</h4>

S와 Q를 같은 값으로 한다. S가 범위에서 랜덤으로 뽑는 방식일 경우 중간값으로 Q를 정한다(ex. S=[256;512], Q=384). 네트워크가 깊어질수록 error가 줄어들고, 훈련 데이터에 scale jittering(가변적인 S 사용하는 것)도 효과가 있는 것으로 보인다.

<p align="center"><img src="{{ img_path }}_figure3.png?raw=true" width="80%"></p>

<h4>4.2 Multi Scale Evaluation</h4>

테스트에 대해서 scale jittering이 진행된다. Q = {S - 32, S, S + 32} 혹은 {S_min, 0.5(S_min + S_max), S_max}를 가진다.

<p align="center"><img src="{{ img_path }}_figure4.png?raw=true" width="80%"></p>

<h4>4.3 Multi Crop Evaluation</h4>

pass

<h4>4.4 ConvNet Fusion</h4>

pass

<h4>4.5 Comparison Woth The State Of The Art</h4>

<p align="center"><img src="{{ img_path }}_figure5.png?raw=true" width="80%"></p>

---

<h3>5. Conclusion</h3>

네트워크의 깊이가 성능에 미치는 영향을 잘 보여주었고, 큰 filter를 사용하는 것보다 작은 filter들을 쌓는 것이 더 분류에 도움이 된다는 것도 잘 설명했다.

중간에 multi-crop이네 하는 것이 있었는데 딱히 정리할 필요는 없다고 생각해서 다 짤랐다.