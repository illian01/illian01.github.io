---
layout: post
title: Perceptron
tag: [DeepLearning]
---

{% assign img_path = site.assets_path | append: site.DeepLearning | append: "/" | append: page.title | append: "/" | append: page.title %}

1950년대 후반에 Rosenblatt라는 사람이 Perceptron이라는 것을 제안했다.

인간의 신경세포(neuron)은 이렇게 생겼다고 한다.

<p align="center"><img src="{{ img_path }}_figure1.png?raw=true" width="100%"></p>

Dendrite(가지돌기)에서 신호를 받고, Cell body(세포체)가 하나의 신호로 만든 다음 Axon(축삭)을 타고 간다. 라는 뉴런의 신호 흐름을 본딴 것이 Perceptron이 되겠다.

<p align="center"><img src="{{ img_path }}_figure2.png?raw=true" width="100%"></p>

$$ out = f \left( \sum_{k=1}^{n} x_{k} \cdot w_{k} \right) $$라고 표현할 수 있다. $$ x_{1} \sim x_{n} $$들이 입력으로 들어오면 각각에 weight 값이 곱해져서 다 더한다. 여기에 편향값(bias)를 더하고 activation function에 입력으로 주면 output을 얻을 수 있다. activation function은 여기서 step function이 사용되었는데,
입력이 어느 이상이면 1, 아니라면 0을 리턴하는 함수다.

이 Perceptron을 이용해서 선형적인 문제를 해결 할 수 있다. AND 와 OR문제를 예시로 생각해보자. 입력은 0과 1의 중의 값으로 2개가 들어온다고 했을 때 가능한 data point를 알 수 있다. 

<p align="center"><img src="{{ img_path }}_figure3.png?raw=true" width="100%"></p>

여기서 빨간 선이 한개의 perceptron이라고 생각하면 된다. weight와 bias 값을 이용해 직선을 그어 구분을 하는 것이다. 이 직선의 위인지 아래인지(output이 1인지 0인지) activation function이 말해줄 수 있을 것이다. 만일 입력과 weight가 다차원 벡터라면 다차원 공간의 선형 분류가 가능할 것이다.

하지만 10년 정도 지난 후에 Perceptron이 사장되어 버리는 연구가 발표된다. Minsky와 Papert라는 두 사람이 Perceptron은 비선형 적인 문제를 해결할 수 없다고 수식적으로 증명한 것이다. 여기서 비선형 문제라 함은 대표적으로 XOR문제와 같은 것을 말한다.

<p align="center"><img src="{{ img_path }}_figure4.png?raw=true" width="50%"></p>

도저히 선 하나 그어서는 빨간 포인트들만 구별해 낼 수가 없다. 물론 아예 방법이 없는 것은 아니다. Minsky와 Papert는 Multi-layer Perceptron을 구축하면 비선형 문제를 해결할 수 있다고 말했다. 직선 하나로 안되면 여러개 그어서 비선형을 해결하면 되는 것이다.

<p align="center"><img src="{{ img_path }}_figure5.png?raw=true" width="100%"></p>

그럼 된거 아닌가? 싶었지만 사장된 이유는 바로 이를 학습시킬 방법이 없었기 때문이다. 이제는 당시의 이야기가 되어버렸지만. MLP에서 x가 들어오는 층을 input layer, 가운데 있는 것이 hidden layer, 마지막에 값을 내놓는 것이 output layer가 된다. Minsky는 hidden layer의 parameter들을 학습시킬 방법이 없다고 단정지어 한창 뜨겁던 관심은 사그러지게 된다.
