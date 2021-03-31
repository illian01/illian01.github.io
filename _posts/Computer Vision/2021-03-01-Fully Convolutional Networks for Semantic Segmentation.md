---
layout: post
title: Fully Convolutional Networks for Semantic Segmentation
tag: [ComputerVision, Segmentation]
---

{% assign img_path = site.assets_path | append: site.ComputerVision | append: "/" | append: page.title | append: "/" | append: page.title %}

<h3>1. Introduction</h3>

<p align="center"><img src="{{ img_path }}_figure1.png?raw=true" width="60%"></p>

---

<h3>2. Related work</h3>

pass

---
<h3>3. Fully convolutional networks</h3>

convnet의 component들은(convolution, pooling, activation function) local input region을 가지고, 이는 각 픽셀의 좌표에 연관된다. 
어떤 레이어에서 $$y_{i,j}$$는 

$$ \mathbb{y} _{i, j} = f_{ks} (\{ \mathbb{x}_{si + \delta i, sj + \delta j} \}_{0 \le \delta i, \delta j \le k }) $$

로 정의된다. k는 kernel size, s는 stride이며, $$f_{ks}$$는 matrix multiplicaiton, pooling, activation의 layer의 타입을 나타낸다.

합성함수가 될 때에도 형태가 유지되며 kernel size와 stride는 다음과 같이 유지된다.

$$ f_{ks} \circ g_{k^{'} s^{'}} = (f \circ g)_{k^{'} + (k-1)s^{'}, ss^{'}} $$

이러한 형태의 연산을 하는 레이어만 가진 네트워크를 fully convolutional network라 한다. 
FCN은 어떠한 입력 크기에도 작동하며, 입력에 상응하는 크기(spatial dimension)의 출력을 내놓는다.


<h4>3.1 Adapting classifiers for dense prediction</h4>

전형적인 Classification 모델들은 고정된 형태의 입력을 받아서 non-spatial 한 출력을 내놓는다.
fully connected layer 때문에 고정된 차원 값이 필요하고 공간적인 정보를 날려버리기 때문인데,
이 fully connected layer는 convolution의 한 형태로 볼 수 있어서 convolution layer로 대체할 수 있다.
fc층을 conv층으로 변환시킴으로 크기가 자유로운 입력과 출력을 얻을 수 있게 된다.

<p align="center"><img src="{{ img_path }}_figure2.png?raw=true" width="80%"></p>

출력이 공간적 정보를 담은 맵을 가지게 되었으니 ground truth를 가지고 forward와 backward가 쭈욱 이어지게 되었다.

<h4>3.2 Shift-and-stitch is filter rarefaction</h4>

shift-and-stitch 라는 트릭이 있는데, 안쓴다.

pass

<h4>3.3 Upsampling is backwards strided convolution</h4>

upsampling을 bilinear와 같이 interpolation하는 방법이 있지만, 여기서는 deconvolution하는 방법을 제안한다(transposed convolution이 맞는 말이지만).
upsample이 convolution layer로 구성된다면 네트워크 안에서 end-to-end로 학습할 수 있게 되고, 이 레이어가 쌓인다면 복잡한 비선형 학습이 가능할 것이다.

<h4>3.4 Patchwise training is loss sampling</h4>

patch로 학습하는 게 전체 이미지를 사용하는 것과 큰 차이가 없다고 한다. 자세한 것은 뒤에 설명된다.

---

<h3>4. Segmentation Architecture</h3>

ILSVRC에 나왔던 분류기들을 in-network upsampling과 skip connection을 추가해서 dense prediction을 할 수 있도록 수정한다.
픽셀 간 multinomial logistic loss를 사용하며, metric은 mean pixel intersection over union을 모든 클래스 평균내서 사용한다.
여기서 train과 validation은 PASCAL VOC 2011 segmentation challenge를 사용한다.

<h4>4.1 From classifier to dense FCN</h4>

분류기를 convolutionalizing해서 segmentation 모델로 바꿔본다. 모델은 AlexNet, VGG16, GoogLeNet을 사용한다.
VGG16은 VGG19와 별다른 차이가 없어서 선택되었다. GoogLeNet은 보조 분류기를 제하고 마지막 출력만 사용하며 마지막에 있는 average pooling을 떼고 사용한다.

모든 모델의 fully connected layer를 떼어내고 convolution layer로 변환시킨다.
PASCAL VOC 2011의 클래스 개수에 맞게 1x1 convolution으로 채널 수를 맞추고, bilinear로 upsampling 한다.

fine-tune 한 결과가 아래 표인데, VGG16을 사용한 모델의 점수가 당시 sota를 뛰어넘은 점수라고 한다.

|  | FCN-AlexNet | FCN-VGG16 | FCN-GoogLeNet |
|---:|:---:|:---:|:---:|
|mean IU|39.8|**56.0**|42.5|
|forward time|50 ms|210 ms|59 ms|
|conv.layers|8|16|22|
|parameters|57M|134M|6M|
|rf size|355|404|907|
|max stride|32|32|32|

<h4>4.2 Combining what and where</h4>

아래 그림과 같이 feature hierarchy 를 조합해서 공간적인 정보가 좀 더 정밀한 output을 만드는 Fully Convolutional Net을 만든다.

<p align="center"><img src="{{ img_path }}_figure3.png?raw=true" width="100%"></p>

Section 4.1처럼 기존의 classifier를 segmentation 모델로 finetune은 했는데, 결과물을 눈으로 보면 많이 뭉개져있다. 
총 stride가 32나 되기 때문인데, 디테일한 정보를 많이 잃어버려 upsample을 했을 때 만족스럽지 못한 것이다. 

<p align="center"><img src="{{ img_path }}_figure4.png?raw=true" width="80%"></p>

이를 해결하기 위해서 skip connection 들을 추가한다. 얕은 층의 feature map은 작은 receptive field를 가지지만 동시에 작은 stride를 가진다.
따라서 깊은 층과 얕은 층의 feature map을 합치면 모델은 전체 구조를 살피면서 작은 부분의 prediction까지 가능하게 된다.

첫번째로, pool4에서 나온 feature map을 1x1 conv 로 클래스 수에 맞게 prediction을 한다.
그리고 마지막 층인 conv7의 feature map을 2x upsampling layer에 통과시켜서 16 stride짜리 feature map을 만든다.
이렇게 나온 두 feature map을 더해서 원본 크기로 upsample하면 된다. 이렇게 만들어진 모델을 FCN-16s라 한다.

FCN-16s를 원본 이미지 크기가 아니라 2x upsample한 후, 이번엔 pool3를 1x1 conv prediction 한 뒤 더하는 과정을 한번 더 한다. 
이 결과물을 원본 이미지 크기로 upsample하면 이게 FCN-8s이다.

여기서 2x upsampling layer는 bilinear interpolation 으로 초기화 하되, 학습할 수 있도록 설정한다.

|  | pixel acc. | mean acc. | mean IU | f.w. IU |
|---:|:---:|:---:|:---:|:---:|
|FCN-32s-fixed|83.0|59.7|45.4|72.0|
|FCN-32s|89.1|73.3|59.4|81.4|
|FCN-16s|90.0|75.7|62.4|83.0|
|FCN-8s|**90.3**|**75.9**|**62.7**|**83.2**|

<h4>4.3 Experimental framework</h4>

**Optimization** SGD, mini batch 20, momentum 0.9, learning rate ($$10^{-3}$$, $$10^{-4}$$, $$10^{-5}$$)
for (AlexNet, VGG16, GoogLeNet), weight decay $$5^{-4}$$ or $$2^{-4}$$, doubled lr for biases,
zero-init the class scoring layer, dropout is same with original nets.

**Fine-tuning**  pass

**More Training Data** pass

**Patch Sampling** patch를 구해서 학습하는 것과 이미지를 통째로 입력으로 쓰는 것은 수렴하는데 큰 차이가 없다. 
수렴은 똑같은데 수렴까지 가는데 시간이 달라서 전체 이미지를 사용하는 것이 더 좋다고 한다.

<p align="center"><img src="{{ img_path }}_figure5.png?raw=true" width="60%"></p>

**Class Balancing** class balancing은 필요 없다고 한다.

**Dense Prediction** 마지막 deconvolution filter는 bilinear로 고정된다. 
그 외 중간에 등장하는 upsampling은 bilinear로 초기화 하되 학습할 수 있도록 한다.

**Augmentation** train data에 random mirroring과 약간의 jittering이 들어갔다고 하나, 큰 변화는 없었다고 한다.

**Implementation** Caffe로 구현되었으며 하나의 NVIDIA Tesla K40c 에서 train/test가 진행되었다. 
코드는 [http://fcn.berkeleyvision.org](http://fcn.berkeleyvision.org)에 공개되어 있다.

---

<h3>5. Results</h3>

PASCAL VOC, NYUDv2, SIFT Flow 데이터셋에 적용한 결과를 다른 모델들과 비교한다. 자세히 쓸 필요는 없을 듯 하다.

pass

---

<h3>6. Conclusion</h3>

Fully Convolutional 구조로 classification net을 segmentation net으로 바꾸고,
multi-resolution layer를 섞어서 큰 성능 향상을 가져왔다.