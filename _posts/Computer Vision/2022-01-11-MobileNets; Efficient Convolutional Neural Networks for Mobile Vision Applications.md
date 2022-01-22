---
layout: post
title: MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications
tag: [ComputerVision, ModelArchitecture]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" %}

<h3>arXiv, 2017</h3>

---

<h3>1. Introduction</h3>

이전 모델이 너무 무거워서 모바일 단말에서는 사용하지 못하는 문제를 해결하고자 MobileNet을 제안한다.

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure1.png" width="100%"></p>

---

<h3>2. Prior work</h3>

pass

---

<h3>3. MobileNet architecture</h3>

<h4>3.1. Depthwise separable convolution</h4>

일반적인 convolution 층은 $$D_{F} \times D_{F} \times M$$을 입력으로 받고,
$$D_{F} \times D_{F} \times D_{N}$$을 출력한다(패딩이 적용될 때).
이 convolution 층의 커널 크기가 $$D_{K}$$라고 했을 때, 파라미터 수는 $$D_{K} \times D_{K} \times M \times N$$가 된다.
이 커널은 모든 픽셀 위치에서 적용되고, 따라서, standard convolution의 computational cost는 다음과 같다.

$$D_{K} \cdot D_{K} \cdot M \cdot N \cdot D_{F} \cdot D_{F}$$

이 standard convolution은 2단계로 분해될 수 있고,
이 분해된 convolution을 depthwise separable convolution이라 한다.
Depthwise separable convolution은 depthwise convolution과 pointwise convolution으로 구성된다.

Depthwise convolution은 각 convolution 필터가 입력 특징의 각 채널에 적용되어 convolution 연산이 적용되는 방법이다.
즉, 입력 특징이 $$M$$개의 채널을 가지고 있을 때, 필터의 갯수도 $$M$$개가 되어서 각 채널에 $$D_{K} \times D_{K}$$의 필터가 적용된다.

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure2.png" width="80%"></p>

따라서, depthwise convolution의 파라미터 수는 $$D_{K} \times D_{K} \times M$$이고, computational cost는 다음과 같다.

$$D_{K} \cdot D_{K} \cdot M \cdot D_{F} \cdot D_{F}$$

Pointwise convolution은 standard convolution의 커널 크기가 1x1인 경우를 말한다.
따라서, 파라미터 수는 $$M \times N$$이고, computational cost는 다음과 같다.

$$M \cdot N \cdot D_{F} \cdot D_{F}$$

두 텀을 더하면 depthwise separable convolution의 computational cost를 얻을 수 있다.

$$D_{K} \cdot D_{K} \cdot M \cdot D_{F} \cdot D_{F} + M \cdot N \cdot D_{F} \cdot D_{F}$$

Standard separable convolution을 depthwise separable convolution으로 바꿀 때 얻게 되는 연산량의 이득은 다음과 같다.

$$\frac{D_{K} \cdot D_{K} \cdot M \cdot D_{F} \cdot D_{F} + M \cdot N \cdot D_{F} \cdot D_{F}}{D_{K} \cdot D_{K} \cdot M \cdot N \cdot D_{F} \cdot D_{F}} = \frac{1}{N} + \frac{1}{D^{2}_{K}}$$

이에 따르면, $$3 \times 3$$커널을 사용했을 때, 8~9배의 연산량 이득을 얻을 수 있다.

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure3.png" width="60%"></p>

<h4>3.2. Network structure and training</h4>

MobileNet 구조는 다음과 같다.

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure4.png" width="60%"></p>

끝단의 FC를 제외한 모든 Conv 층에는 batch normalization과 ReLU가 적용된다.
성능 비교를 위해 이 모델을 standard convolution으로 구현한 것은 다음과 같이 대응해 치환되는 방식으로 구성된다.

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure5.png" width="50%"></p>

<h4>3.3. Width multiplier: Thinner models</h4>

하이퍼 파라미터 $$\alpha$$를 사용해 간단히 모델을 작게 만드는 방법을 소개한다.
$$\alpha \in (0,1]$$의 범위를 가지며, 각 conv 층의 출력 채널에 일괄적으로 $$\alpha$$를 곱해서 모델 크기를 줄이는 방식이다.
적용된 후의 연산량은 다음과 같다.

$$D_{K} \cdot D_{K} \cdot \alpha M \cdot D_{F} \cdot D_{F} + \alpha M \cdot \alpha N \cdot D_{F} \cdot D_{F}$$

<h4>3.4. Resolution multiplier: Reduced representation</h4>

하이퍼 파라미터 $$\rho$$를 사용해 간단히 모델을 작게 만드는 방법을 소개한다.
$$\rho \in (0, 1]$$의 범위를 가지며, 입력 해상도를 $$\rho$$만큼 줄이는 방식이다.
$$\alpha$$와 $$\rho$$가 둘다 적용된다면 각 conv층의 연산량은 다음과 같다.

$$D_{K} \cdot D_{K} \cdot \alpha  M \cdot \rho D_{F} \cdot \rho D_{F} + \alpha M \cdot \alpha N \cdot \rho D_{F} \cdot \rho D_{F}$$

뒤에 실험에서는 "$$\alpha$$ MobileNet-(입력 해상도)" 형식으로 표기해 어떤 하이퍼 파라미터가 쓰였는지 알 수 있다.

---

<h3>4. Experiments</h3>

<h4>4.1. Model choices</h4>

Standard convolution과 depthwise convolution의 차이

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure6.png" width="60%"></p>

Shallow한 버전은 MobileNet에서 입출력 사이즈가 $$14 \times 14 \times 512$$인 층 5개를 날린 것이다.
층을 아예 날리는 것보다, 전체 층의 채널 수를 줄이는 게 유리하다는 결과이다.

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure7.png" width="60%"></p>

<h4>4.2. Model shrinking hyperparameters</h4>

하이퍼 파라미터 $$\alpha$$에 대한 변화

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure8.png" width="60%"></p>

하이퍼 파리미터 $$\rho$$에 대한 변화

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure9.png" width="60%"></p>

GoogleNet, VGG16과 비교

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure10.png" width="60%"></p>

Squeezenet, AlexNet과 비교

<p align="center"><img src="/assets/Computer Vision/MobileNets; Efficient Convolutional Neural Networks for Mobile Vision Applications/figure11.png" width="60%"></p>

<h4>4.3. Fine grained recognition</h4>

뒤의 실험들은 다양한 task에 대해서 잘 적용된다는걸 보인다.

pass

<h4>4.4. Large scale geolocalization</h4>

pass

<h4>4.5. Face attributes</h4>

pass

<h4>4.6. Object detection</h4>

pass

<h4>4.7. Face embedding</h4>

pass

---

<h3>5. Conclusion</h3>

pass
