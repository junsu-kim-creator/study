import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque

from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.datasets import mnist

# 상태가 입력, 큐함수가 출력인 인공신경망 생성
def build_net(state_size, action_size):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(512, input_dim=state_size, activation='relu',
                        kernel_initializer=RandomUniform(-1e-3,1e-3)))
    model.add(tf.keras.layers.Dense(512, input_dim=state_size, activation='relu',
                        kernel_initializer=RandomUniform(-1e-3,1e-3)))
    model.add(tf.keras.layers.Dense(512, input_dim=state_size, activation='relu',
                        kernel_initializer=RandomUniform(-1e-3,1e-3)))
    model.add(tf.keras.layers.Dense(512, input_dim=state_size, activation='relu',
                        kernel_initializer=RandomUniform(-1e-3,1e-3)))
    model.add(tf.keras.layers.Dense(512, input_dim=state_size, activation='relu',
                        kernel_initializer=RandomUniform(-1e-3,1e-3)))
    model.add(tf.keras.layers.Dense(action_size))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['acc'])
    print(model.summary())
    return model
    
    def Replay_Bellman(main_model, target_model, batch, discnt_rate):
    
    states = np.vstack([x[0] for x in batch])
    actions = np.array([x[1] for x in batch])
    rewards = np.array([x[2] for x in batch])
    states2 = np.vstack([x[3] for x in batch])
    done = np.array([x[4] for x in batch])
    
    # 벨만 최적 방정식을 이용한 예측 Q 값
    Q = rewards + discnt_rate * np.max(target_model.predict(states2), axis=1) * ~done

    target = main_model.predict(states)
    target[np.arange(len(states)), actions] = Q

    return states, target
    
    mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

ds_train = tf.data.Dataset.from_tensor_slices((
    x_train, y_train)).shuffle(10000).batch(10000)

images,labels = [],[]
for image, label in ds_train:
    images = image 
    labels = label
    break

def batch_out(data):
    for image, label in data:
        return image.numpy(), label.numpy()

image, label = batch_out(ds_train)

state_size = (28,28,1)
#state_size = 28*28
action_size = 10

Episodes = 5000
discnt_rate = 0.99
batch_size = 64
Re_m = 20000
Target_update_frequency = 5
e_min = 0.01

# 리플레이 메모리, 최대 크기 Re_m = 20000
Re_buffer = deque(maxlen=Re_m)
scores = []  # 점수 기록
ep,loss,accurancy = [],[],[]

G = tf.Graph()
with G.as_default():
    # 메인 모델과 타깃 모델 생성
    main_model = build_net(state_size, action_size)
    target_model = build_net(state_size, action_size)
    
    # 타깃 모델 초기화
    target_model.set_weights(main_model.get_weights())

    for episode in range(Episodes):
        e = 1. / ((episode / 10) + 1)  # 입실론
        if e < e_min:
            e = e_min
        done = True
        score = 0
        ep.append(episode)

        # env 초기화
        state = image[episode]/255
        #state = np.reshape(state, [1, state_size])
        state = np.reshape(state, (1, 28, 28, 1))

        for i in range(3):
            # Q-net에서 입실론 탐욕 정책으로 현재 상태에 따른 행동 선택

            if np.random.rand() <= e:
                action = random.randrange(action_size)
            else:
                action = np.argmax(main_model.predict(state))

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            reward = (action == label[episode])

            # 리플레이 메모리에 경험 샘플 [s, a, r, s', d] 저장
            Re_buffer.append((state, action, reward, state, done))
            # 리플레이 메모리가 배치 크기만큼 쌓이면, 무작위로 추출한 배치로 학습
            if len(Re_buffer) > batch_size:
                batch = random.sample(Re_buffer, batch_size)
                states, target = Replay_Bellman(main_model, target_model, batch, discnt_rate)
                his = main_model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

            # 일정 주기마다 타겟 모델 초기화
            if score % Target_update_frequency == 0:
                target_model.set_weights(main_model.get_weights())
            
            score += reward

        loss.append(his.history['loss'][0])
        accurancy.append(his.history['acc'][0])
        scores.append(score)
        print("Episode: {:>4}, Score: {:>4},label: {:>4} Re_len: {:>4}, e: {:>4.2f}".format(episode, score, label[episode] ,len(Re_buffer), e))

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(ep,loss, 'y', label='train loss')

acc_ax.plot(ep,accurancy, 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
#plt.plot(scores)
