import random
import numpy as np
from collections import deque
from keras import Sequential, layers
from keras.optimizers import Adam


class DQLearn:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate, epsilon greedy
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.96
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))

        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse', metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
            return int(random.uniform(0, self.action_size))
        else:
            act_vals = self.model.predict(state)
            return np.argmax(act_vals[0])

    def replay(self, batchsize):
        minibatch = ra ndom.sample(self.memory, batchsize)

        for state, action, reward, next_state, terminal in minibatch:

            # if finished, make target as reward
            target = reward

            if not terminal:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))

            # Q(s, a)
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
