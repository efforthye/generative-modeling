# 케라스에서 sequential 모델이나 함수형 api를 사용해 신경망 구조 정의 가능
from tensorflow.keras import layers, models
from utils import draw_neural_network

# 아래와 같이 01_keras_MLP_sequential_model.py 와 동일한 모델을 만들 수 있다.
# Input 층: 네트워크의 시작점으로 네트워크가 기대하는 입력 데이터 크기를 튜플로 정의해 준다.
# Input 층에 임의의 이미지 개수를 전달할 수 있기 때문에 배치 크기는 필요하지 않다.
input_layer = layers.Input(shape=(32, 32, 3))

# Flatten 층: 입력을 하나의 벡터로 펼친다. (여기서는 input_layer 객체를 함수처럼 호출하여 사용함)
# 길이: 3072(32*32*3)
# 펼치는 이유: 뒤따르는 Dense 층이 다차원 배열이 아니라 평평한 입력을 기대하기 때문이다.
# 다른 종류의 층은 입력으로 다차원 배열을 사용해야 한다.
x = layers.Flatten()(input_layer)

# Dense 층: 기본적인 신경망 구성 요소이다. 이 층에는 이전 층과 완전하게 연결되는 유닛이 있다.
# 이 층의 각 유닛은 이전 층의 모든 유닛과 연결된다. + 연결마다 하나의 가중치(양수 or 음수)가 동반된다.
# 유닛의 출력: 이전 층에서 받은 입력과 가중치를 곱하여 더한 것이다.
# 그 다음 비선형 활성화 함수를 통과하여 다음 층으로 전달된다.
# 이처럼 활성화 함수는 신경망이 복잡한 함수를 학습하는데 중요한 역할을 한다.
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
