# 케라스에서 sequential 모델이나 함수형 api를 사용해 신경망 구조 정의 가능
from tensorflow.keras import layers, models
from utils import draw_neural_network

# Sequential 클래스를 사용하여 MLP 모델 정의
# sequential 모델은 일렬로 층을 쌓은 네트워크를 빠르게 만들 때 사용하기 좋다.
# 하지만 Sequential 모델보다는 함수형 API를 사용하는 것이 좋다. 
# (신경망의 구조가 점점 복잡해짐에 따라 장기적으로 함수형 API를 사용하면 
# 심층 신경망의 설계를 자유롭게 할 수 있다.)
model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(200, activation='relu'),
    layers.Dense(150, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 모델 구조 출력
model.summary()

# 시각화
layer_sizes = [32*32*3, 200, 150, 10]
layer_names = ['Input\n(Flatten)', 'Dense\n(ReLU)', 'Dense\n(ReLU)', 'Output\n(Softmax)']
draw_neural_network(layer_sizes, layer_names, save_path='output/01_keras_MLP_sequential_model.png')
