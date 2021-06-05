# 03.Machine-learning-on-XPS-spectrum
Repository for my undergraduate paper for Machine Learning analysis of XPS spectrum

## Introduction

![XPS configuration](https://drive.google.com/uc?export=view&id=1Q6co995-kS0TN0WC-PQJa0JnVWoAMhvb)

X-ray photoelectron spectroscopy(XPS)는 물질의 표면 특성을 연구하기 위한 장비이다. 특정한 Beam Source로부터 방출되는 전자기파를 시료의 표면에 입사하면, 표면의 전자들은 전자기파의 진동수에 비례하는 에너지를 받게 된다. 에너지를 받은 입자는 excitation 되어 물질 표면의 Work function을 제외한 만큼의 운동 에너지를 갖고 외부로 방출된다. 이렇게 방출되는 전자를 외부의 detector로 포집하여 kinetic energy 또는 binding energy에 따른 intensity를 측정할 수 있다. 결과적으로 물질 표면에 존재하는 여러 결합 특성을 파악할 수 있게 된다.

본 연구에서는 XPS Spectrum Analysis를 자동화하기 위하여 1차원 Convolution Neural Network (CNN)을 사용하였다. 모델을 훈련하기 위하여 우선적으로 대규모의 구성 파라미터를 생성하여, 이로부터 가상의 XPS 스펙트럼 데이터를 생성하였다. Spectrum fitting을 위한 model function에는 Pseudo Voigt function을 사용하였으며, 훈련된 모델은 입력으로 받아들인 XPS 스펙트럼의 Peak position, Width, Amplitude 등을 Overlapping area에도 불구하고 상당히 잘 예측함을 확인할 수 있었다.

<br>

# Deep Learning Applied to Peak Fitting of Spectroscopic Data in Frequency Domain

## Hyeong Seon Park · Seong-Heum Park · Hyunbok Lee · Heung-Sik Kim∗

Department of Physics, Kangwon National University, Chuncheon 24341, Korea

## Abstract

A data-driven study of material properties and functional materials design based on it requires
high-throughput and comparative analyses of the results of experimental spectroscopy with those
from first-principles electronic structure calculations. Hence, an efficient machine-learning-based
computational tool to extract electronic structure information from experimental data without
human intervention is in high demand. Here, we test the capability of deep neural network
models to fit photoemission spectroscopy (PES) data in the frequency domain with unknown PES
peak positions, numbers, and widths. A one-dimensional convolution neural network (CNN) was
employed in combination with fully connected layers (FCL), and the trained model was applied
to photoemission spectra for the sulfur 2p states in poly(3-hexylthiophene) (P3HT) molecules and
oxygen 1s states in indium tin oxide (ITO). We conclude by further discussing potential ways to
improve the performance of the model.

Keywords: Photoemission spectroscopy, Machine learning, Deep neural network

