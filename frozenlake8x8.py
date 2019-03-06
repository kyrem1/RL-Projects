import numpy as np
import gym

env = gym.make('Determinis  ntic-FrozenLake-v0')

# Hyperparameters
alpha = 0.9
gamma = 0.9
epsilon = 0.2
maxstep = env.observation_space.n

print("Observation Space: ", env.observation_space)
print("Actions: ", env.action_space, "\n\n")

Q_table = np.zeros((env.observation_space.n, env.action_space.n))


# Eplison Greedy
def choose_action(state, t):
    if np.random.uniform(0, 1) < epsilon :
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[state, :])
    return action


def learnPred(state, state2, reward, action):
    predict = Q_table[state, action]
    target = reward + gamma * np.max(Q_table[state2, :])
    Q_table[state, action] = Q_table[state, action] + alpha * (target - predict)


# Q(state,action) ← (1−α) Q(state,action) + α(reward+ γ * max_a Q(next state, all actions))
def learn(state, state2, reward, action):
    Q_table[state, action] = (1 - alpha) * Q_table[state, action] + alpha * (reward + gamma * np.max(Q_table[state2, :]))


def eval(Q):
    penalties, successes = 0, 0
    plays = 100

    for i in range(1, plays):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state, :])
            state2, reward, done, info = env.step(action)
            state = state2
            if reward == 0.0 and done:
                penalties = penalties + 1
    successes = plays - penalties
    print("S: ", successes, "F: ", penalties)
    return successes


def train():
    for i in range(1, 3000):
        state = env.reset()
        done = False
        while not done:
            action = choose_action(state, i)
            state2, reward, done, info = env.step(action)
            if i % 2 == 0:
                learnPred(state, state2, reward, action)
            else:
                learn(state, state2, reward, action)
            state = state2

        if i % 100 == 0:
            print(f"Episode: {i}")

            eval(Q_table)

    print(Q_table)


train()
eval(Q_table)
pass
