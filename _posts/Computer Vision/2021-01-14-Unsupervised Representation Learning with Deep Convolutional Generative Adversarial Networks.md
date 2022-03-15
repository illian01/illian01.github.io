---
layout: post
title: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
tag: [ComputerVision, GAN]
---

{% assign img_path = site.assets_path | append: site.Papers | append: "/" | append: page.title | append: "/" %}

<h3>ICLR, 2015</h3>

---

<h3>1. Introduction</h3>

GAN은 학습이 매우 불안정하고, 이해할 수 없는 출력을 자주 낸다.
이러한 GAN의 학습을 이해하고 시각화 하는 연구는 많지 않다. 그에 대한 연구로 이 논문의 주요 내용은 다음과 같다.
1. Deep Convolutional GANs(DCGAN)을 제안하고 constraints에 대해서 이야기한다. DCGAN은 대부분의 설정에서 안정적인 학습이 되었다.
2. 이미지 분류에 미리 학습된 discriminator를 사용한다. 여타 비지도학습에 비해서도 괜찮은 성능을 보였다.
3. filter들을 시각화해서 보인다. 그 중 특정한 object를 그리기 위해 학습된 필터들을 경험적으로 보인다.
4. 생성자가 재미있는 벡터 산술적 특성을 가지고 있음을 보인다. 이를 통해서 간단한 의미적인 조작이 가능함을 보인다.

---

<h3>2. Related work</h3>

pass

---

<h3>3. Approach and model architecture</h3>

GAN 생성물의 해상도를 개선하려는 시도들이 있었으나 신통치 않았다. 
GAN에 CNN을 적용해서 이를 해결하려고 했고, 많은 실험 끝에 다양한 데이터셋 에서도 안정적으로 학습하는 모델의 형식을 찾아냈다.
1. discriminator의 모든 conv net은 pooling을 사용하지 않고 strided convolution을 사용한다. downsampling도 스스로 학습하도록 하는 것인데,
generator에 대해서도 같은 접근을 해서 스스로 upsampling을 학습 할 수 있도록 fractional-strided convolution을 사용한다.
2. 트렌드에 따라 fully connected layer를 지우고 global average pooling을 사용한다. 모델의 안정성은 개선되나 수렴 속도가 느려졌다.
generator의 첫번째 layer는 fully-connected이나, output은 4x4를 가지도록 reshape해서 사용한다
3. generator와 discriminator에 batch normalization을 적용한다. 모든 레이어에 직접적으로 적용하면 sample oscillation과 model instability를 불러일으킨다.
이는 generator의 output layer와 discriminator의 input layer에는 적용하지 않음으로 피할 수 있다.
4. generator의 activation은 ReLU를 사용한다. 마지막 출력에는 Tanh를 사용한다.
5. discriminator의 activation은 LeakyReLU를 사용한다.

<p align="center"><img src="{{ img_path }}figure1.png?raw=true" width="100%"></p>

---

<h3>4. Details of adversarial training</h3>

Large-scale Scene Understanding(LSUN), Imagenet-1k, Faces 3개 데이터셋에 대해서 학습을 했다.
- 이미지는 [-1,1]의 값으로 scaling된다.
- Adam(lr=0.0002, beta_1=0.5), batch_size=128.(모멘텀 beta 0.9는 불안정 하다고 함)
- 모든 weight는 mean=0, std=0.02인 표준분포에서 초기화된다.
- LeakyReLU의 기울기는 0.2를 사용한다.

<h4>4.1 LSUN</h4>

고해상도 이미지 LSUN 데이터셋에 대해 학습한다. 300만 이상의 샘플을 가지고 있다.
1 epoch 후의 생성 이미지들과 5 epoch 후의 생성 이미지들을 보여주며 단순히 overfitting/memorization을 통해서 이미지를 생성하는 것이 아니라고 주장하는데, 
1 epoch에서 작은 lr과 mini-batch sgd를 사용해서 한번에 data를 다 기억할 수 있을 것 같지는 않다는 것과,
5 epoch에서 이미지에 나타나는 noise들이 under-fitting의 증거이기 때문이라고 한다.

<p align="center"><img src="{{ img_path }}figure2.png?raw=true" width="80%"></p>

<p align="center"><img src="{{ img_path }}figure3.png?raw=true" width="80%"></p>

<h4>4.1.1 Deduplication</h4>

generator가 데이터를 기억해버리는 가능성을 좀 더 줄이기 위해 de-duplication process를 진행한다.
약 275,000 개의 중복이 제거된다.

<h4>4.2 Faces</h4>

dbpedia에서 얻은 이름들을 가지고 랜덤하게 웹상에서 가져온 얼굴들을 사용한다.
이 데이터셋에는 10K 사람에 대한 3M개 이미지가 있다. OpenCV의 face detector를 돌린 결과 35만 개의 face box를 얻을 수 있었다.
이 face box들을 학습에 사용한다.

<h4>4.3 Imagenet-1k</h4>

32x32 min-resized center crop을 사용한다.

---

<h3>5. Empirial validation of DCGANs capabilities</h3>

<h4>5.1 Classifying CIFAR-10 using GANs as a feature extractor</h4>

unsupervised representation learning algorithm을 평가하기 위해서 이를 feature extractor로 사용하는 방법이 있다.
supervised dataset에서 평가하려는 모델로 특징을 뽑아내고 linear model을 붙여서 나온 결과를 평가한다.

baseline은 4800 짜리 feature map을 내놓는 레이어에 K-means를 붙인 것이다. 
측정하려는 DCGAN의 경우 Imagenet-1k에 대해서 학습한다.
discriminator의 모든 레이어에서 특징맵을 가져와 이를 max-pooling해서 4x4 grid로 만든 뒤 flatten, concatenate 해서 28672 길이의 벡터로 만들고,
뒤에 regularized linear L2-SVM이 붙어서 학습된다. Exemplar CNN보다는 낮은 성능이나 K-means 기반의 다른 모델보다는 뛰어난 성능을 보인다.
또한, CIFAR-10에 대해서는 학습한 적이 없으나 좋은 성능을 보여주므로 robustness또한 증명된다.

<p align="center"><img src="{{ img_path }}figure4.png?raw=true" width="80%"></p>

<h4>5.2 Classifying SVHN digit using GANs as a feature extractor</h4>

<p align="center"><img src="{{ img_path }}figure5.png?raw=true" width="50%"></p>

---

<h3>6. Investigating and visualizing the internals of the networks</h3>

<h4>6.1 Walking in the latent space</h4>

manifold 위를 걸어다니면서(?) 보는 것은 memorization과 같은 현상에 대한 힌트를 얻을 수 있다.
이미지의 의미가 갑작스럽게 뒤바뀌어 버린다면 모델이 데이터를 기억만 하고 있다고 추론할 수 있다.
혹은 latent space 의 변화가 의미적인 변화를 일으킨다면 모델은 표현을 학습 했다고 생각할 수 있다.

<p align="center"><img src="{{ img_path }}figure6.png?raw=true" width="80%"></p>

<h4>6.2 Visualizing the discriminator features</h4>

guided backpropagation을 활용해여 이미지를 복구한다. 
discriminator가 침실의 전형적인 구조에 포함되는 침대, 창문 등을 학습한 것을 볼 수 있다.

<p align="center"><img src="{{ img_path }}figure7.png?raw=true" width="80%"></p>

<h4>6.3 Manipulating the generator representation</h4>

<h4>6.3.1 Forgetting to draw certain objects</h4>

discriminator가 무엇을 학습하는지 봤으니 generator가 무엇을 학습했는지도 보려한다.
그 방법으로 창문을 generator에서 삭제하는 실험을 한다.

끝에서 두 번째 convnet에서(the second highest conv layer?) 창문에 해당하는 모든 특징 맵을 삭제한 뒤,
이미지를 생성하면 모델은 창문을 그리는 법을 잊고 다른 객체로 그 자리를 채우는 것을 볼 수 있다. 

<p align="center"><img src="{{ img_path }}figure8.png?raw=true" width="80%"></p>

<h4>6.3.2 Vector arithmetic on face samples</h4>

vector("king") - vector("man") + vector("woman")이라는 산술 연산을 통해서 Queen을 뽑아낼 수 있는지 확인한다.
1개의 샘플을 활용하는 것은 불안정해서 3개의 Z 벡터를 평균하는 것으로 사용한다. 이 산술 연산이 꽤나 잘 들어맞는 걸 볼 수 있다.

<p align="center"><img src="{{ img_path }}figure9.png?raw=true" width="80%"></p>
*한개의 벡터에 +-0.25 로 uniform noise를 섞은 것 8개를 생성해서 총 9개씩 뽑은 것이다.*

<p align="center"><img src="{{ img_path }}figure10.png?raw=true" width="80%"></p>
*왼쪽에 사람 -> 오른쪽에 사람 으로 변화하는 과정*

---

<h3>7. Conclusion and future work</h3>

pass
