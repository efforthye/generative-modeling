# 케라스에서 sequential 모델이나 함수형 api를 사용해 신경망 구조 정의 가능
from tensorflow.keras import layers, models
from utils import draw_neural_network

# 아래와 같이 01_keras_MLP_sequential_model.py 와 동일한 모델을 만들 수 있다.
input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(input_layer)
x = layers.Dense(units=200, activation='relu')(x)
x = layers.Dense(units=150, activation='relu')(x)
output_layer = layers.Dense(units=10, activation='softmax')(x)
model = models.Model(input_layer, output_layer)

# 모델 구조 출력
model.summary()

# 시각화
layer_sizes = [32*32*3, 200, 150, 10]
layer_names = ['Input\n(Flatten)', 'Dense\n(ReLU)', 'Dense\n(ReLU)', 'Output\n(Softmax)']
draw_neural_network(layer_sizes, layer_names, save_path='output/02_keras_MLP_functional_api.png')
