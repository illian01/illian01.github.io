---
layout: post
title: Backpropagation
tag: [DeepLearning]
---

Perceptron 이론이 관짝에 들어가고 빙하기가 꽤 지났는데, 이 이론을 다시 관짝에서 끌어낸 것이 Hinton의 Backpropagation(역전파)이다.
이를 통해서 hidden layer의 weight들을 학습할 수 있다는 것을 알게 되었고, 좀 더 복잡한 문제 역시 해결이 가능함을 보였다.

Backprop에 대해서 알기 전에 Gradient descent부터 짚고 가자.

Gradient descent(경사 하강법)은 근삿값을 찾는 최적화 알고리즘이다.
함수의 기울기를 따라서 가장 낮은 곳을 향해 이동시키는 것을 반복한다.

$$ f(x) $$ 라는 함수에서 초기값을 $$ x_{0} $$, $$ x_{i} $$의 다음 값을 $$ x_{i+1} $$라고 한다.
이 때, $$ x_{i+1} $$값은 다음과 같이 정의된다.

$$ x_{i+1} = x_{i} - \gamma \Delta f(x_{i}) $$

$$ \gamma $$는 어느정도의 보폭으로 함수의 낮은 곳을 찾아 갈 것인지 정하는 값이 된다.

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_figure1.png?raw=true" width="50%"></p>

가장 낮은 지점에서의 값을 $$ x_{opt} $$라고 할때, 이보다 큰 값일때의 미분값은 양수, 
작을 때의 미분 값은 음수가 되어 위 식에 따라 최적값을 찾아간다.

---

고등학교 때 열심히 외우던 미분공식 중에 chain rule이 있다. 함수에 인자가 함수로 들어왔을 때 이름을 F라 하면,
이 F를 미분하는 공식은 다음처럼 정리된다.

$$
\begin{align*}
F(x)     &= f(g(x)) \\
F^{'}(x) &= f^{'}(g(x)) \cdot g^{'}(x)
\end{align*}
$$

여기서 $$ t = g(x) $$로 둔다면, $$ f^{'}(g(x)) = \frac{dy}{dt} $$, $$ g^{'}(x) = \frac{dt}{dx} $$

$$ F^{'}(x) = \frac{dy}{dx} = \frac{dy}{dt} \cdot \frac{dt}{dx} $$

Backpropagation 알고리즘의 핵심은 이게 전부다. 이후의 자세한 설명은 cs231n의 설명을 가져왔다.

$$ f(x, y, z) = (x+y)z $$라는 함수가 있고, Computational graph를 그려보자.

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_figure2.png?raw=true" width="80%"></p>

$$ x = -2, y = 5, z = -4 $$일 때, $$ f(x, y, z) $$ 는 -12의 값을 가진다. 자 여기 예제에서 구하고자 하는 것은 
$$ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} $$ 값들이다. 

$$ q = x + y $$라고 정의하면 $$ \frac{\partial f}{\partial z} = q = 3 $$라고 쓸 수 있다. 
동시에 $$ \frac{\partial f}{\partial q} = z $$이다. chain rule에 따라서 
$$ \frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial x} $$로 쓸 수 있다.
$$ z $$ 값은 정해져 있고, $$ \frac{\partial q}{\partial x} $$는 1이므로 $$ \frac{\partial f}{\partial x} $$는 -4가 된다.
같은 이유로 $$ \frac{\partial f}{\partial y} $$ 역시 -4이다.

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_figure3.png?raw=true" width="80%"></p>

이런 식으로 함수들의 연속에서 gradient는 전파 될 수 있다. 
$$ \frac{\partial f}{\partial x} = \frac{\partial f}{\partial \alpha} \frac{\partial \alpha}{\partial \beta} \cdots 
\frac{\partial \psi}{\partial \omega} \frac{\partial \omega}{\partial x}$$와 같이 복잡하더라도, 
각 node에서는 해당 node 출력값에 자기 변수로 미분한 값을 propagation되어 온 값에다 곱하면 출력값에 해당하는 gradient를 얻을 수 있다.

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_figure4.png?raw=true" width="80%"></p>

Perceptron 에서 소개할 때는 step function을 쓴다고 했는데, 이 함수는 0에서는 미분이 불가능하고, 나머지는 해도 0인 값이라
Backprop을 적용할 수 없다. 따라서 앞으로는 미분 가능한 함수를 activation function으로 쓴다.
$$ \sigma $$는 sigmoid function으로, S자 형태 안에서 0~1사이의 값을 반환한다.

$$ \sigma (x) = \frac{1}{1+e ^{-x}}, \sigma^{'}(x) = \sigma (x) (1 - \sigma (x)) $$

$$y$$가 출력이면 W의 gradient는 $$ x*(y*(1-y)) $$가 되겠다. x와 W는 vector의 형태를 가지고
내적하는 형태인데, 어차피 각 weight에 대해서 편미분하면 같은 요소에 해당하는 x만 남아서 똑같다고 보면 된다.

---

Backpropagation 알고리즘이 MLP(Multi-Layer Perceptron)을 학습시킬 수 있다는 것을 보이긴 했지만, 아직 문제가 남아있었다.

vanishing gradient - 위의 sigmoid 함수의 도함수를 그려보면 최댓값이 0.25가 나오는데, 이는 layer를 하나 지날 때마다 gradient 전파가 0.25 배씩 줄어든다는 것을 의미한다. 한 두번은 전파가 되겠지만, layer가 쭉 쌓이다보면 전파되는 값이 가파르게 줄어들어 0에 수렴해가게 된다. 이렇게 gradient가 전파되지 못하면 weight의 학습이 불가능해진다. 물론 vanishing gradient의 문제가 sigmoid 함수에만 있는 것은 아니다.

weight initialization - weight의 초기값을 어떻게 설정하는지는 학습에 큰 영향을 준다. 출력 값이 치우치게 돼서 gradient가 사라질 수도 있고, 단순히 랜덤하게 설정했다가 운이 없어서 local minimum에 빠지기 쉽다. 

이외에도 deep learning이 날개를 펴기까지는 더 많은 이론들이 필요하지만 하나씩 알아가봐야겠다.