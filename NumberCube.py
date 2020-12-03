import random
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from collections import deque

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

class NumberCube:
    def __init__(self, goal):
        self.goal = goal.flatten()
        self.board = self.goal.copy() 
        self.row, self.col = np.shape(goal)
        self.state_size = self.row * self.col
    
    def left(self, idx): 
        if idx % self.col != 0:
            self.board[idx], self.board[idx-1] = self.board[idx-1], self.board[idx]

    def up(self, idx):
        if idx >= self.row:
            self.board[idx], self.board[idx-3] = self.board[idx-3], self.board[idx]

    def right(self, idx):
        if idx % self.col != (self.row - 1):
            self.board[idx], self.board[idx+1] = self.board[idx+1], self.board[idx]
        
    def down(self, idx):
        if idx <= (self.state_size - self.row - 1) :
            self.board[idx], self.board[idx+self.row] = self.board[idx+self.row], self.board[idx]
    
    def take_action(self,action):
        idx = np.where(self.board == 0)[0][0]
        if action == 0: self.left(idx) 
        elif action == 1: self.up(idx)
        elif action == 2: self.right(idx)
        else: self.down(idx)

    def step(self,action):
        self.take_action(action)
        reward = (np.sum(self.board == self.goal.flatten()) - 8) / 9
        done = np.array_equal(self.board, self.goal.flatten())
        return self.board, reward, done

    def reset(self):
        self.board = self.goal.copy()
        [self.take_action(i) for i in [np.random.randint(4) for i in range(np.random.randint(10)+1)]]
        return self.board
        
# 상태가 입력, 큐함수가 출력인 인공신경망 생성
def build_net(state_size, action_size):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(590, input_dim=state_size, activation='relu',
                        kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(590, input_dim=state_size, activation='relu',
                        kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(590, input_dim=state_size, activation='relu',
                        kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(590, input_dim=state_size, activation='relu',
                        kernel_initializer='he_uniform'))
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
    
goal = np.array([[1, 2, 3],[4, 0, 6],[7, 8, 9]])
env = NumberCube(goal)
state_size = env.state_size
action_size = 4

Episodes = 10000
discnt_rate = 0.99
batch_size = 64
Re_m = 20000
Target_update_frequency = 5
counters = []
loss,accurancy = [],[]
e_min = 0.01

# 리플레이 메모리, 최대 크기 Re_m = 20000
Re_buffer = deque(maxlen=Re_m)

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
        done = False

        count = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state,[1,state_size]) / 9

        while not done:
            # env.render()  # 행동 모니터링
            count += 1 

            # Q-net에서 입실론 탐욕 정책으로 현재 상태에 따른 행동 선택

            if np.random.rand() <= e:
                action = random.randrange(action_size)
            else:
                action = np.argmax(main_model.predict(state))
            
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            state2, reward, done = env.step(action)
            state2 = np.reshape(state2,[1,state_size]) / 9

            # 리플레이 메모리에 경험 샘플 [s, a, r, s', d] 저장
            Re_buffer.append((state, action, reward, state2, done))
            # 리플레이 메모리가 배치 크기만큼 쌓이면, 무작위로 추출한 배치로 학습
            if len(Re_buffer) > batch_size:
                batch = random.sample(Re_buffer, batch_size)
                states, target = Replay_Bellman(main_model, target_model, batch, discnt_rate)
                hist = main_model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

            # 일정 주기마다 타겟 모델 초기화
            if episode % Target_update_frequency == 0:
                target_model.set_weights(main_model.get_weights())
            
            if count == 15: break
            
            state = state2
        loss.append(hist.history['loss'][0])
        accurancy.append(hist.history['acc'][0])
        counters.append(count)
        #print("Episode: {:>4}, count: {:>4}, Re_len: {:>4}, e: {:>4.2f}".format(episode, count, len(Re_buffer), e))
        print('episode:',episode,'memory:',len(Re_buffer),'count:',count)

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(loss, 'y', label='train loss')

acc_ax.plot(accurancy, 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
#plt.plot(counters)
