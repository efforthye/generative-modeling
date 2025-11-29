# 실행 방법
# cd ~/Desktop/programs/study/ai/generative-modeling
# source myenv/bin/activate
# python src/CIFAR-10_dataset_preprocessing.py
import numpy as np
from tensorflow.keras import datasets, utils

# CIFAR-10 데이터셋 로드
# x_train과 x_test는 각각 [50000, 32, 32, 3]와 [10000, 32, 32, 3] 크기의 넘파이 배열이다.
# y_train과 y_test는 각각 [50000, 1]과 [10000, 1]의 넘파이 배열로
# 각 이미지의 클래스에 대해 0~9 범위의 정수 레이블을 담는다.
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

NUM_CLASSES = 10

# 픽셀 채널 값이 0과 1 사이가 되도록 이미지의 스케일을 조정한다.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 레이블을 원한 인코딩한다.
# y_train과 y_test의 크기는 각각 [50000, 10]과 [10000, 10]이 된다.
y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)

# 훈련 이미지 데이터(x_train)가 [50000, 32, 32, 3]크기의 텐서 형태로 저장된다.
# (열이나 행이 없는 4차원 텐서이다. 텐서는 다차원 배열로서 행렬을 2차원 이상으로 확장한 것이다.)

# 결과 확인
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"x_train 값 범위: {x_train.min()} ~ {x_train.max()}")
# x_train shape: (50000, 32, 32, 3)
# x_test shape: (10000, 32, 32, 3)
# y_train shape: (50000, 10)
# y_test shape: (10000, 10)
# x_train 값 범위: 0.0 ~ 1.0
