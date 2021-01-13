---
layout: post
title: Activation function
tag: [DeepLearning]
---

딥러닝에서 Activation function(활성함수)의 존재는 매우 중요하다.

이전 층에서 들어온 값을 잘 합쳐서 어떻게 다음층으로 넘겨줄지는 이 함수가 결정한다. Perceptron 에서 사용된 step function을 생각해본다면 어떤 data x가 들어왔을 때, 이 x가 '선 위에있어요' 혹은 '선 아래있어요'를 판단해주는 역할을 하는 것이다.

대부분의 활성함수는 비선형성을 가진다. Identity 라고 이름 붙여진 직선 $$ f(x)=x $$는 대표적인 선형함수다. 간단한 MLP에서 모든 node는 Identity 활성함수를 가진다고 생각해보자.

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_mlp.png?raw=true" width="60%"></p>

$$
\begin{align*}
& (w_{1}x_{1} + w_{2}x_{2} + b_{1})w_{5} + (w_{3}x_{1} + w_{4}x_{2} + b_{2})w_{6} + b_{3}
\\ =& (w_{1}w_{5} + w_{3}w_{6})x_{1} + (w_{2}w_{5} + w_{4}w_{6})x_{2} + w_{5}b_{1} + w_{6}b_{2} + b_{3}
\\ =& a_{1}x_{1} + a_{2}x_{2} + b
\end{align*}
$$

위와같이 결국 하나의 Perceptron을 사용하는 것과 다름이 없고, 비선형적인 문제를 풀 수 없게 된다.

---

<!---------------------------------------------------------------------------------------->
<h3>1. Step function</h3>

$$  
f(x) = 
\begin{cases}
1 & x > 0 \\
0 & \mathrm{otherwise}
\end{cases}
$$

```
def step(x):
    y = x.copy()
    y[y >= 0] = 1
    y[y != 1] = 0
    return y

x = np.linspace(-100, 100, 10000)
y = step(x)

plt.plot(x, y)
plt.show()
```

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_step.png?raw=true" width="50%"></p>

0과 1만을 반환하는 함수다. 미분이 불가능하다 == backprop이 불가능하다.
<!---------------------------------------------------------------------------------------->

<!---------------------------------------------------------------------------------------->
<h3>2. Sigmoid function</h3>

$$ f(x) = \frac{1}{1+e^{-x}} $$

```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-8, 8, 10000)
y = sigmoid(x)

plt.plot(x, y)
plt.show()
```

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_sigmoid.png?raw=true" width="50%"></p>

0~1의 값을 반환하는 함수이다. 도함수의 최대값은 0.25로 layer가 깊어지면 backprop때 gradient가 소실된다.
거기다가 반환 값의 평균이 0.5라 layer가 지날수록 분산이 커져 학습이 제대로 진행되지 않도록 만들 수 있다.
<!---------------------------------------------------------------------------------------->

<!---------------------------------------------------------------------------------------->
<h3>3. tanh function</h3>

$$ f(x) = \frac{sinh(x)}{cosh(x)} = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} = \frac{e^{2x}-1}{e^{2x}+1}$$

```
x = np.linspace(-5, 5, 10000)
y = np.tanh(x)

plt.plot(x, y)
plt.show()
```

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_tanh.png?raw=true" width="50%"></p>

tanh가 -1~1 값을 반환해서 중심 값이 0이 되어 sigmoid의 문제점 중 하나는 해결된다.
exp로 인한 깊은 layer에서의 gradient 소실은 여전하다.
<!---------------------------------------------------------------------------------------->

<!---------------------------------------------------------------------------------------->
<h3>4. ReLU(Rectified Linear Unit) function</h3>

$$  
f(x) = max(x, 0) = 
\begin{cases}
x & x > 0 \\
0 & \mathrm{otherwise}
\end{cases}
$$

```
def relu(x):
    y = x.copy()
    y[y < 0] = 0
    return y

x = np.linspace(-100, 100, 10000)
y = relu(x)

plt.plot(x, y)
plt.show()
```

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_relu.png?raw=true" width="50%"></p>

모르겠으면 일단 ReLU를 쓰면 된다. 연산이 매우 빠르다. input이 양수면 그대로 출력하기 때문에, gradient가 1이 되어서 gradient 소실을 막는다.
input이 음수면 값을 이후로 전파시키지 않는다.

노드가 음수 값을 출력하면 gradient는 0이 된다. 가중치를 업데이트 할 수가 없게 되는데, 이 노드들이 다시 살아나는 경우는 별로 없다. 이 현상을 dying relu라 한다.
<!---------------------------------------------------------------------------------------->

<!---------------------------------------------------------------------------------------->
<h3>5. Leaky ReLU function</h3>

$$
f(x) = max(0.01x, x) = 
\begin{cases}
x & x > 0 \\
0.01x & \mathrm{otherwise}
\end{cases}
$$

```
def leaky_relu(x):
    y = x.copy()
    y[y < 0] *= 0.01
    return y
```

dying relu를 해결하기 위한 방안으로, 값이 음수일 떄 0.01이라는 작은 값을 곱해서 보낸다. gradient를 미세하게 흘려서(미분값이 0이 되지 않게 해서) 해당 node가 죽지 않게 하는 것이다.
<!---------------------------------------------------------------------------------------->

<!---------------------------------------------------------------------------------------->
<h3>6. Parametric ReLU function</h3>

$$  
f(x) = max(\alpha x, x) = 
\begin{cases}
x & x > 0 \\
\alpha x & \mathrm{otherwise}
\end{cases}
, (\alpha < 1)
$$

```
def prelu(x, alpha=0.2):
    y = x.copy()
    y[y < 0] *= alpha
    return y

x = np.linspace(-100, 100, 10000)
y = prelu(x)

plt.plot(x, y)
plt.show()
```

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_prelu.png?raw=true" width="50%"></p>

Leaky ReLU라고 생각하자. $$ \alpha $$라는 값이 주어진 것 뿐이다.
<!---------------------------------------------------------------------------------------->

<!---------------------------------------------------------------------------------------->
<h3>7. ELU(Exponential Linear Unit) function</h3>

$$  
f(x) = 
\begin{cases}
x & x > 0 \\
\alpha (e^{x} - 1) & \mathrm{otherwise}
\end{cases}
$$

```
def elu(x, alpha=0.3):
    y = x.copy()
    y[y < 0] = alpha * (np.exp(y[y < 0]) - 1)
    return y

x = np.linspace(-5, 5, 10000)
y = elu(x)

plt.plot(x, y)
plt.show()
```

<p align="center"><img src="{{ site.assets_path }}{{page.title}}/{{page.title}}_elu.png?raw=true" width="50%"></p>

평균 출력 이 0에 가까워진다. $$\alpha$$를 일반적으로 1로 설정하는데, 이는 도함수를 연속적으로 만들어주고 gradient descent의 속도를 높여준다.
<!---------------------------------------------------------------------------------------->