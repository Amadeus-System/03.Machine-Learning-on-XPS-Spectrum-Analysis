# 03.Machine-learning-on-XPS-spectrum
Repository for my undergraduate paper for Machine learning analysis of XPS spectrum

https://lelouch0316.tistory.com/entry/Iterative-Peak-Fitting-of-Frequency-Domain-Data-via-Deep-Convolution-Neural-Networks?category=1076241

* **Date : 2022-01-02**
* **Last Modified At : 2022-06-27**

# Iterative Peak-Fitting of Frequency-Domain Data via Deep Convolution Neural Networks

<br>
<p align="center" style="color:gray">
    <img src="https://i0.wp.com/xpslibrary.com/wp-content/uploads/XPS-with-polarized-x-rays.png?fit=1192%2C772&ssl=1">
</p>
<center><b></b></center>
<br>

<!-- TOC -->

- [Iterative Peak-Fitting of Frequency-Domain Data via Deep Convolution Neural Networks](#iterative-peak-fitting-of-frequency-domain-data-via-deep-convolution-neural-networks)
    - [Introduction](#introduction)
    - [Methods](#methods)
        - [Description of Neural Network Architectures](#description-of-neural-network-architectures)
        - [Generation of Training, Validation, and Test Datasets](#generation-of-training-validation-and-test-datasets)
        - [Training and Validation of Neural Networks](#training-and-validation-of-neural-networks)
    - [Results](#results)
        - [CNN Training and Validation](#cnn-training-and-validation)
        - [Iterative Peak Subtraction via Trained CNN and Basin-Hopping Optimization](#iterative-peak-subtraction-via-trained-cnn-and-basin-hopping-optimization)
    - [Conclusion](#conclusion)
    - [References](#references)

<!-- /TOC -->

이번 글에서는 연구인턴 시절에 진행했던 XPS 연구의 후속연구를 소개하고자 한다.

이전에 소개했던 1단계 연구는 여러 시행착오로 인해 기대한 만큼의 좋은 성과를 내지 못했고, 별로 높은 수준이 아닌 국내저널에 게재되었다. 반면, 후속연구 논문은 이전 연구에 비해 좋은 성과를 달성하여 낮은 IF의 JKPS 저널에 게재되었다. 이는 본래 진행하려고 했던 ARPES 연구 프로젝트의 2단계에 해당하는 연구였다.

기본적인 연구의 계획은 크게 변하지 않았으나 여러가지 CNN 모델을 직접 리팩토링하고, 다양한 하이퍼파라미터 튜닝을 하는 경험을 쌓을 수 있었다. 특히 마지막에 추가로 도입된 Basin-Hopping 알고리즘은 딥러닝 모델이 감지하지 못하는 Residual Fitting Error를 잘 감지하여 연구를 마무리하는데 큰 도움이 되었다.

이전 글에서 설명한 것처럼, 이 연구는 XPS 실험 데이터에 대한 자동화된 분석과정을 딥러닝 모델을 통해 구현하려는 것이었다. 즉, 숙련된 인간 연구원의 개입이 최소화된 데이터 분석을 다루고자 했다.

블로그에 처음 온 분들을 위해 간단히 설명하자면, 여러 종류의 CNN 모델을 훈련시키고 1차원 Noise가 섞인 Synthetic XPS 데이터를 여러개의 직교되지 않은(Non-Orthogonal) Peak들로 분해하는 연구였다. 주목할 만한 점이 있다면, 어느 시점에서 더이상 성능이 향상되기 어려운 딥러닝 모델의 사용방법을 바꿔 적용함으로써, 더 좋은 결과를 달성했다는 점이다. 

이후 고전적인 Basin-Hopping 알고리즘을 적용하여 불완전하게 남아있는 Fitting Error를 감소시켰다. 6개의 서로 다른 CNN 모델 구조 중에서 우리가 제안했던 Squeeze-and-Excitation 네트워크의 변형모델은 최고의 성능을 보여주었다. 또한 손실함수의 선택에 따른 훈련 성능의 의존성도 논의하였다. 최종적으로 변형된 SENet 모델을 Graphene, $MoS_{2}$, $WS_{2}$와 같은 물질의 실험적인 광전자 스펙트럼 데이터에 적용하였다.



## Introduction

보통 재료과학 분야에서 물성예측 모델을 훈련시키기 위해 축적된 계산데이터를 활용하는 것과는 대조적으로, XPS와 같은 실험적인 분광학 분야에서 머신러닝을 적용하는 것은 상대적으로 덜 연구되어왔다. 

XPS 및 머신러닝의 응용연구 분야에서는, 초기에 실험적인 PTM과 STM 스펙트럼으로부터 유용한 물리적 정보를 추출하려는 몇가지 응용연구가 있었다.

구체적으로는, Core-level XPS 스펙트럼에 대한 딥러닝 응용연구가 있었는데, 딥러닝 모델을 이용하여 주어진 테스트 물질의 화학적 구성요소들의 화학양론을 예측하는 것이 주제였다. 그러나 실제 물질의 구성비와 10% 정도의 오차를 보였고, 이후 관련된 분야의 연구들은 별로 발표되지 않았다.

당시에 연구를 진행하는 과정에서, 실험적인 스펙트럼 데이터를 분석하는 것은 몇가지 어려운 점이 있었다.

첫번째는 스펙트럼을 Non-Orthogonal한 Peak 요소로 분해하는 과정에서 존재하는 내재된 비고유성이다. 즉, 간단히 말하자면 2개 이상의 Peak가 존재하여 Overlapping Area가 발생하는 스펙트럼은, 그러한 영역의 해석에서 단일 솔루션이 존재하기 어려웠다.

두번째로 실험적 스펙트럼 데이터가 매우 제한적이라는 것이었다. 즉, 실제적인 표면 물리현상을 모방하기 위해 XPS 스펙트럼과 유사한 데이터를 합성하여 훈련데이터로 사용했지만, 당연히 가장 좋은 데이터 자원은 실제 연구실에서 축적된 XPS 데이터였을 것이다. 그러나 그러한 Real XPS 데이터는 10년 가까이 XPS 연구를 한 연구실에서도 수백개정도밖에 없었다.

세번째로 전통적인 Spectroscopy 연구분야에서 오랜 경험과 직관을 가진 인간 연구원이 딥러닝 모델에 비해 성능면에서 우월하다는 사실이었다. 즉, 실제로 XPS 실험을 하는 연구원들은 단순히 스펙트럼의 모양만을 갖고 해석하는 것이 아니라 데이터 외적인 여러 표면물리학적 지식과 관련분야에서 이루어진 레퍼런스 연구들의 지식을 종합하여 실험결과를 해석하였다. 예를 들면, XPS 데이터에서 특정한 부분에 Peak가 존재하지 않는 것처럼 보이더라도 숙련된 연구원이라면 특정한 Energy level에서 반드시 특정 원자(예를 들면 Carbon)의 Peak가 존재한다는 사실을 알고 있었다.

추가적으로 XPS 분야에는 이미 많은 분광학적 도구들이 존재했다. 보통 XPS 데이터는 에너지나 진동수 도메인에서 측정되는데 이와 관련된 Raman, Photoemission, 광흡수, 광발광, 핵자기공명, 비탄성 X선, 중성자 산란 등 실험적 스펙트럼 데이터의 분석을 자동화하기 위한 고전적인 계산도구들이 이미 상당히 발전되어 있었다. 우리의 연구는 이 목록에 딥러닝 기반의 방법론을 추가하려는 것이었다.

다음 그림은 연구에서 딥러닝 모델의 응용방식이었던 Iterative Peak Fitting 과정을 보여준다. 이전 단계의 연구에서 우리가 훈련시킨 딥러닝 모델은 여러가지 XPS 관련 정보들을 동등한 수준으로 출력하지는 못했었다. 

즉, 모델이 상당히 정확하게 예측하는 정보와 그렇지 못한 정보가 있었다. 그 중에서 최대높이(최대면적)을 갖는 한 종류의 Peak는 유달리 잘 예측하는 것을 발견했었기 때문에, 모델의 한정된 예측성능만을 반복적인 방식으로 적용함으로써 모델의 성능을 끌어올리려는 시도가 주요한 아이디어였다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/149705030-692e6c1a-f64d-467f-a4ea-11fe5c6b248a.png">
</p>
<center><b>Figure 1. A cartoon illustration of decomposing a noisy spectrum (black curve) with multiple peak features by sequentially applying regression and subtraction of the largest-area peak (red and blue curves) until no peak feature is left</b></center>
<br>



## Methods

### Description of Neural Network Architectures

XPS 스펙트럼으로부터 최대한 내재된 넓이를 갖는 하나의 Peak를 정확하기 추출하기 위해, 잘 설계된 CNN 구조가 필요했다. 이를 위해 여섯종류의 CNN 구조를 도입하여 연구상황에 맞게 변형하고 Peak 추출 성능을 Benchmark 하였다.

도입된 CNN 구조는 다음과 같았다. 모델들의 정렬 순서는 네트워크 구조에서 복잡성이 증가하는 순서이다.

* **LeNet**
  딥러닝을 공부한 사람들은 알겠지만, LeNet은 최초로 제안된 CNN 구조이다. 이미지에 대한 Convolution 연산과 Pooling 연산을 반복하는 것으로 구성된다.
* **Alex-ZFNet**
  Alex-ZFNet은 LeNet의 향상된 버전으로 간주될 수 있다. MaxPooling, Dropout, ReLU 함수를 도입하였다.
* **VGGNet**
  VGGNet은 균등한 네트워크 구조이면서도 컨볼루션 필터의 증대된 숫자를 선택하여 다른 길을 취했다.
* **ResNet**
  ResNet은 소위 말하는 Skip connections를 취한 것이다, 이는 네트워크 레이어 속 깊이까지 손실함수의 back propagation을 강화하기 위해서 고안되었다.
* **SENet**
  SENet에서는 소위 말하는 'Squeeze-and-Excitation (SE) block'이 도입되어, 적은 양의 계산코스트 증가로 성능을 향상시키고자 하였다.
* **m-SENet**

연구에서 최종적으로 m-SENet을 특성추출을 위한 Best Performance 모델로 선택하였다.

다음 그림의 (a), (b)에서 보이는 것처럼 각각의 Sparse-Dense Block은 연속적으로 연결된 합성곱 층과 하나의 SE 블록의 여러 쌍으로 구성된다. 여기서 Identity 맵핑은 하나의 Sparse-Dense 블록 안에서 CONV-CONV-SE의 Triplet 2개 출력층과 입력을 맵핑한다. 또한, 분포를 조정하는 Batch Normalization과 SE 층들은 인접한 Sparse-Dense Block 사이에 놓였는데, 이는 합성곱의 결과로 나오는 특성맵의 가중치를 재조정하기 위해서이다. 이 구조는 극도로 깊은 모델 안에서도 효율적인 학습을 허용하는 것으로 알려져 있다.

마지막으로 회귀(Regression) 층은 Global Average Pooling을 통해 연결되었다. 모델의 총 학습가능한 파라미터의 수는 4183,830개였고, Overlapped Average Pooling은 모든 서브샘플링 과정 안에서 사용되었다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/149705041-ceec83bb-d0b6-459a-bc7b-53ad1bef9d5d.png">
</p>
<center><b>Figure 2. a A schematic representation of m-SENet architecture, where detailed layering structure inside a sparse-dense block is depicted in (b)</b></center>
<br>



### Generation of Training, Validation, and Test Datasets

이제 데이터셋에 대해 이야기해보자.

보통 머신러닝/딥러닝을 처음 공부하게 되면, 이미 준비되어 있는 다양한 예제 데이터들이 학습자를 반겨준다. 하지만 실제로 물리학과 같은 분야에 머신러닝을 적용하려고 하면, 준비되어 있는 데이터는 거의 존재하지 않는다. 데이터는 대부분 파편화되어 인터넷 공간상에 여기저기 흩어져 있고, 데이터를 모으더라도 각기 다른 연구조건에서 이루어진 실험들은 동일한 데이터로 취급되기 어렵다.

그렇기 때문에, 내가 알기로 물리학 분야에서 데이터가 축적되어 온 극히 일부 분야(계산물리학, 입자물리학 등)를 제외하고, XPS와 같은 실험물리학 분야에서는 실질적으로 사용할 수 있는 데이터가 거의 존재하지 않았다.

그럼 데이터가 존재하지 않는데, 어떻게 머신러닝 응용 연구를 할 수 있을까? 보통 이 상황에서 연구자들이 선택하는 것은 현실의 XPS 스펙트럼을 모방할 수 있는 Synthetic Training Data를 만드는 것이다.

우리의 연구에서는 비교적 간단한 방법으로 XPS 스펙트럼 데이터를 재현하려고 했다. 하지만 XPS 스펙트럼의 모양이 다른 물리 데이터들에 비해 간단해 보일지라도, XPS의 실험적 특성을 온전히 간직한 데이터를 생성하는 것은 매우 어려운 일이다. 왜냐하면 본질적으로 물질의 표면에 대한 상호작용 정보가 데이터에 포함되어 있을 것이기 때문이다. 종종 매우 높은 수준의 물리학 저널에 올라오는 머신러닝 응용논문을 보면, 해당분야의 권위자들이 모인 연구그룹에서 높은 수준의 이론물리를 적용하여 극도로 정교하게 고안된 Synthetic 데이터를 갖고 연구를 한 내용이 발견된다. 그만큼 제대로 현실의 물리현상을 반영한 데이터를 만드는 것은 어려운 일이다.

실제 연구에서는 Real XPS 스펙트럼 데이터가 모델의 최종검증에만 사용되어야 할 만큼 매우 부족했다. 그러므로 Pseudo Voigt Function을 적용하여 모델 훈련을 위한 합성 데이터셋을 생성하였다. 여기서 Voigt 함수라는 것은 실험적으로 측정되는 X-ray Diffraction 또는 광전자 데이터로부터 Peak Fitting을 하는데 종종 사용되는 모델함수라고 한다.

함수의 다음 형태는 인공 데이터셋 안의 각각의 Peak를 생성하기 위해 선택되었다.

$$ f(\omega; \omega_0, I_0, \delta) = I_0 ( I_G e^{- \frac {log(2) (\omega - \omega_0)^2}{\delta^2 \omega_G^2}} + I_L \frac {1}{1 + \frac {(\omega - \omega_0)^2}{\delta^2 \omega_L^2}} ), \tag 1  $$

여기서 $\omega$는 스펙트럼 데이터의 진동수(에너지)로 간주될 수 있고, $\omega_0$, $I_0$, $\delta$ 는 각각 임의로 생성된 피크의 위치, 최대높이, 그리고 (차원없는) 너비이다. $\omega_G$, $\omega_L$, $I_G$, 그리고 $I_L$은 각각 $0.510$, $0.441$, $0.7$ 그리고 $0.3$으로 설정되었다. 위의 Pseudo Voigt Function 안에서의 이러한 파라미터 값은 이전의 PES 연구에서 얻어진 실험적 스펙트럼을 Fitting하기 위해 설정되었다.

최종적으로 각각의 1차원 합성데이터에서 최대 5개까지의 임의의 유사 Voigt 함수들이 더해졌다. 그리고 표준편차가 최대세기의 2%가 되는 Gaussian noise를 추가하였다. 또한, 이웃 Peak들의 중심위치들이 최대 $\delta$의 20%보다 가깝게 되지 않도록 제한하였다.

XPS 실험에서 흔히 보이는 Background Signal은 이 연구에서 고려되지 않았지만, 배경신호의 필터링은 단순한 CNN 모델을 통해 가능하다는 것이 이미 선행연구로 보고되어 있었기에 문제가 되지 않았다.



### Training and Validation of Neural Networks

모델과 데이터가 모두 준비된 상태에서 실질적인 훈련에 들어갔다. 연구실에 예산이 부족해서 GPU가 별로 없었기 때문에(..) 모델훈련은 Nvidia RTX 2080Ti GPU를 사용하여 이루어졌다.

중심위치 $\omega_0$, 너비 $\delta$, 최대넓이를 가진 Peak의 세기 $I_0$, 그리고 스펙트럼 안의 Peak의 수를 표적변수로 정해져 있었고, 전체 손실함수는 이러한 개별 물성에 대한 손실함수의 총합으로 정의되었다.

전체 손실함수 안에서 서로 다른 손실함수 성분들의 기여도를 정규화하기 위해 1, 10, 20, 그리고 2의 가중치들이 $\omega_0$, $\delta$, $I_0$, 그리고 Peak 수를 최적화하는 손실함수의 각 부분에 할당되었고, Adam Optimizer가 사용되었다. SGD 또한 대조군으로 테스트되었으나 성능은 거의 유사했다.

활성화 함수는 모두 LeakyReLU가 사용되었고, 512의 Batch Size와 50 Epoch로 학습이 이루어졌으며, 학습률은 20번째와 40번째 Epoch 이후 Overfitting을 피하기 위해 10%까지 감소되었다. 또한 Dropout 레이어는 사용되지 않았다. (이 부분에 대해서는 제 1저자가 왜 Dropout 레이어를 사용하지 않았는지 약간 의문이다. 아래 결과에서 보겠지만, Overfitting을 억제하는 것이 연구에서 중요한 문제로 보이고, Dropout은 과대적합을 억제하는 매우 효과적인 수단이기 때문이다. 그러나 나는 2저자였기 때문에 무슨 의도가 있는가 싶어서 크게 이의를 제기하지 않았다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/149705044-7d8b40f8-5680-4c45-a78f-e6e66ed582fe.png">
</p>
<center><b>Figure 3. Two examples of synthetically generated data with multiple (up to 5) pseudo-Voigt profiles. Colored and black lines depict each pseudo-Voigt peaks and addition of all peaks with gaussian noise, respectively</b></center>
<br>

최종적으로 CNN Benchmarking을 위해, $1.5 \times 10^6$ 개의 합성 스펙트럼을 생성하였고, 그 중에서 $1.2 \times 10^6$, $1.5 \times 10^5$, $1.5 \times 10^5$ 데이터는 훈련/검증/테스트를 위해 각각 사용되었다. 나중에는 데이터셋의 크기에 따른 성능향상 확인을 위해 데이터셋의 규모를 $10^7$으로 증가하였다.


## Results

### CNN Training and Validation

다음 그림은 서로 다른 CNN 모델들의 훈련 및 검증과정에서의 MSE 손실함수의 변화를 보여준다. 4개의 목표변수들 사이에서 (특별히 중요한) Peak 중심위치의 손실함수가 같이 그려졌다.

Peak의 위치는 특별히 가장 큰 오차를 만드는 경향이 있었다. 이는 실제 XPS 분석에서 Peak의 에너지 레벨을 추정하는 일이 가장 중요하다는 사실과 대응되는 것 같았다.

훈련 및 검증단계의 손실함수의 양상을 보면, 확실히 초창기 CNN 모델이었던 LeNet에서 m-SENet으로 갈수록 점진적인 손실의 감소가 일어나는 것이 명확하게 보인다. 이는 Computer Vision 분야에서 CNN 모델의 발전이 XPS와 같은 Task-Specific한 분야에서도 (약간의 수정이 있으면) 충분히 안정적인 성능을 보일 수 있음을 증명하는 것이었다. 특히 m-SENet은 Peak의 감지와 전반적인 Fitting 성능에 있어서 다른 모델보다 더 나은 표현력(Representation Power)을 보여주었다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/149705045-308737c9-e7ba-4968-8c35-972f6dc8df11.png">
</p>
<center><b>Figure 4. Total (upper row) and peak center loss functions (lower row) of six different CNN architectures, with respect to training epochs. Note that learning rate was reduced to 10% after 20th and 40th epochs. Mean-square error loss function was employed. Left and right columns show training and validation losses, respectively</b></center>
<br>

위 그림에서 보이는 훈련 및 검증 결과들은 MSE를 손실함수로 계산했을 때의 결과이다. 일반적으로 MSE는 회귀문제에서 모델을 직접적으로 학습하기 위한 손실함수로 자주 채택된다. 그러나 몇몇 경우에 MSE는 매우 작은 이상치(Outlier)의 효과를 과대평가하는 경향이 있었다. 이는 모델의 저조한 성능으로 이어질 수 있기에 종종 MAE가 더 나은 선택일 수도 있었다. 흥미롭게도 우리의 Peak Fitting 문제는 그러한 유형의 것으로 발견되었다.

즉, 모델의 학습/검증 과정에서 MSE/MAE 손실함수의 효과를 비교하기 위해, 동등한 환경에서 각각의 CNN 모델을 분리하고 MSE/MAE를 사용하여 각각 훈련시켰다. 이후 훈련된 모델을 테스트 데이터의 Fitting에 적용하고 MAE 손실값을 다시 계산하였다. 다음 그림 Figure 5는 그 결과를 보여준다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/149705046-e3144065-cbe0-49f5-b466-b4894d37179d.png">
</p>
<center><b>Figure 5. MAE loss functions— a center, b number of peaks, c width, and d amplitude of the largest-area peak—of six different CNN architectures on the same test dataset. Blue and orange columns show results employing MSE and MAE
losses during the network training and validation, respectively</b></center>
<br>

그림에서 4개의 목표변수 모두의 테스트 손실값은 모든 CNN 구조에서 MAE가 더 낮은 값을 보여주었다. 해당 결과로부터 MAE 훈련손실함수를 가진 SENet과 m-SENet이 이번 연구에서 거의 동등한 성능을 보인다는 것을 알 수 있었다. 

(사실 이 부분은 논문에 거의 똑같이 설명되어 있지만, 개인적으로 해석이 조금 이상하다고 생각한다. MSE와 MAE는 회귀오차를 계산하는 서로 다른 방법일 뿐이고, 오차 범위를 고려하면 손실함수의 성능과 무관하게 당연히 저런 결과가 나올 것 같다. 더구나 MAE가 MSE보다 회귀학습에서 뛰어난 성능을 보이는 경우는 많지 않다. 오래전에 Gauss가 증명했기 때문이다. 논문 1저자가 이러한 결과를 보여주었을 때, 분명 이상하다고 생각했고 이 부분에 이의를 제기했었다. 그러나 어째서인지 최종논문에 반영되지 않았고, 결국 JKPS 이전에 게재를 시도했던 보다 높은 수준의 두 저널에서 모두 Reject을 받았다. Reject을 받을 때 리뷰어의 Comment도 같이 첨부되어 있었는데, 내가 이상하다고 생각했던 부분이 그대로 지적되고 있었다. 나중에야 이런 일이 있었다는 것을 지도교수님께 말씀드렸지만 그때는 이미 늦은 뒤였다. ㅎㅎ;)

다시 연구 이야기로 돌아가자. 

m-SENet이 Benchmark에서 최고의 성능을 보여주는 것을 확인한 후, 모델의 성능을 강화하기 위해 전체 데이터셋의 크기를 1.5에서 10 million scale로 증가시켰다. 

아래의 Table 1은 1.5와 10 million 데이터셋으로 훈련된 m-SENet의 손실함수 결과를 비교해서 보여준다. 더 큰 훈련 데이터셋을 적용했을 때 명확한 성능 향상이 관찰되었다. 다만, 계산자원의 한계로 인해 모델성능의 포화를 훈련 데이터셋 크기의 함수로 테스트할 수는 없었다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/175870179-e1724bc6-24da-4be0-9dc2-e414a22de950.PNG">
</p>
<center><b>Table 1</b></center>
<br>



### Iterative Peak Subtraction via Trained CNN and Basin-Hopping Optimization

CNN Benchmark 단계가 끝난 이후, XPS 스펙트럼의 연속적인 특성을 추출하는 단계로 넘어갔다. 다음 그림 6은 연구에서 고안된 "Iterative Peak Fitting and Extraction" 전략을 보여준다.

특성을 추출할 때 발생할 수 있는 잠재적인 문제는, Peak Fitting의 아주 작은 Error조차도 스펙트럼 안의 부정적인 특성을 야기할 수 있다는 것이었다. 또한 애초에 유의미한 Peak가 아닐수도 있었다.

이 문제를 해결하기 위해 다음과 같은 방법을 사용하였다.

1. 초기 스펙트럼에서 전체 Peak의 수를 정확하게 결정하고 정확히 그 숫자만큼만 Peak를 추출하기
2. 추출 이후 Negative Intensity를 제거하기 (Figure 6의 왼쪽 아래 패널)

Peak의 수를 예측하는 정확도는 상당히 높기 때문에, 대부분의 경우 추가적인 가짜 Peak를 추출하는 것을 방해할 수 있었다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/149705047-4cbecd9f-b5d3-4434-873d-129ee92ad318.png">
</p>
<center><b>Figure 6. An illustration of iterative peak fitting and subtraction scheme we employed in this study. In the upper left and lower right panels, red and blue curves depict CNN-fitted and exact peak features respectively. Negative intensity caused by subtraction of inaccurate peaks (upper right panel, below the red dashed line) is truncated before applying subsequent peak fitting and subtraction</b></center>
<br>

주어진 하나의 스펙트럼에서 모든 Peak 특성들을 추출한 후, 고전적인 전역최적화 방법인 Basin-Hopping 알고리즘을 사용하여 Fitting 결과를 더욱 최적화하고 남은 Error를 최소 수준까지 감소시켰다.

여기서 Basin-Hopping 알고리즘은 향상된 Simulated Annealing 방법으로 생각할 수 있다. 즉, 각 Iteration에서의 Accept/Reject 결정 이전 단계에서 추가적인 국소적 최적화 단계를 포함하는 알고리즘이다. 이에 대해서는 다른 글에서 자세히 설명하도록 하겠다.

다음 그림은 몇가지 테스트 데이터에 대해 최적화된 결과를 보여준다. (a)에서 반복적인 CNN의 응용은 꽤 좋은 결과를 보여주었다. (b)에서 CNN은 목표 스펙트럼과 상대적으로 나쁜 일치를 보이고 있다. 이 경우 Basin-Hopping 알고리즘이 추가적으로 작동하여 Residual Error을 감소시켰다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/149705048-8a66630d-93f5-4da9-af07-2386a68f024d.png">
</p>
<center><b>Figure 7. Comparison between peak fitting results before and after applying basin-hopping algorithm. Top (a, b) and bottom (c, d) rows show two different cases where the initial iterative CNN-fitting yield good and relative poor agreements with the target spectra, respectively, after which application of basinhopping algorithm reduces any residual errors to almost zero</b></center>
<br>

최종적으로 m-SENet을 실제 XPS 샘플에 적용하였고 예측결과를 실제와 비교하였다. 다음 그림 8을 보자.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/149705038-2dd70aab-6e5f-4b90-9ef0-2079371b8f94.png">
</p>
<center><b>Figure 8. Human- (upper row) and machine-fitted (lower row) photoemission spectra of a graphene C 1s, b MoS2 Mo 3d, c MoS2 S 2p, d WS2 S 2p, and e WS2 W 4f. In each panel black, red, and green curves represent raw data with background signal subtracted, fitted spectrum, and residual signal, respectively, where fitted peaks are depicted as colored filled curves</b></center>
<br>

XPS 연구를 하는 사람들 입장에서는, 모델의 정교함이 그렇게 대단하지는 않다고 말할 수 있을 것 같다. 그러나 우리가 연구에서 보이고자 했던 것은 오랜 시간 XPS 연구를 수행한 연구자들의 도메인지식이 개입되지 않고도, 딥러닝 모델의 패턴인식만으로 XPS 스펙트럼에 대한 특성추출이 어느 정도까지 자동화될 수 있는가를 확인하려고 했던 것이므로 나름대로 만족할 만한 성과가 나왔다.

다음 그림은 모델의 예측성능과 인간 연구원의 분석결과를 대조한 것이다.

<br>
<p align="center" style="color:gray">
    <img src="https://user-images.githubusercontent.com/76824867/175870175-7deb1bce-0210-4df9-a574-8d81d942330b.PNG">
</p>
<center><b>Table 2</b></center>
<br>



## Conclusion

연구에서 다소 아쉬운 점이 있다면 Synthetic Data의 생성과정에서, 표면현상에 대한 물리적 지식을 포함시키지 않고 임의로 선택한 범위 안에서 균등한 Random Parameter로 생성하였다는 것이다. 아마 내가 물리를 조금 더 잘했다면(..), 조금 더 실제현상을 모방하는 데이터를 만들 수 있었을지도 모른다. 또한, 우리가 사용한 Pseudo Voigt Function 대신 다른 모델함수를 적용할 수도 있었을 것 같다. 보다 효과적인 XPS 모델함수를 사용했다면, Fitting 과정에서의 Fake Peak의 발생을 억제할 수도 있었을 것이다.

연구를 진행하는 동안, 운동량 도메인에서의 스펙트럼 데이터를 분석하는 CNN의 잠재적인 성능을 경험할 수 있었다. 1차로 진행되었으나 실패에 가까웠던 선행연구와 비교하면, 2차 연구에서는 모델의 Peak Fitting 성능이 상당히 발전되었다. 특히 진보된 CNN 모델의 적용과 Iterative한 방식으로 작동하는 특성추출의 응용이 상당히 잘 작동된 것 같다.

정리하자면, 우리가 제안한 m-SENet 모델은 다른 CNN 모델들과 비교하여 더 우월한 성능을 보여주었다. 또한 Fitting Error를 최소화하는 과정에서 Basin-Hopping과 같은 전역최적화 알고리즘의 도움도 컸다. 

결과적으로 딥러닝 기반의 XPS 스펙트럼 분석은 잘 훈련된 연구원의 도메인지식 기반의 Peak Fitting과 비교할만한 성과를 내었다. 이러한 성과는 미래에 최소한의 인간지식의 개입을 가진 대규모 스펙트럼 분석에 응용될 수도 있을 것이다.



## References

* Seong-Heum Park, Hyeongseon Park, Hyunbok Lee, Heung-Sik Kim
J. Korean Phys. Soc. 79(12), 1199 - 1208 (2021)

---
