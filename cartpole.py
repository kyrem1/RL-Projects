import numpy as np
import gym
from collections import deque
from dqlearn import DQLearn

# Deep Q-Learning model
# Input is the action space
# Output is the Q_hat-Values for each action in the action space

# Q(state,action) ← (1−α) * Q(state,action) + α * (reward+ γ * max_a Q(next state, all actions))

episodes = 2000
x_values = []
y_values = []

if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    print(env)
    agent = DQLearn(4, 2)

    scores = deque(maxlen=100)

    for e in range(episodes):

        state = env.reset()
        state = np.reshape(state, [1, 4])

        for time_t in range(500):

            action = agent.act(state)

            next_state, reward, terminal, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            agent.remember(state, action, reward, next_state, terminal)

            state = next_state

            if terminal:
                scores.append(time_t)
                avg_score = 0
                for score in scores:
                    avg_score += score

                avg_score = avg_score / 100
                x_values.append(avg_score)
                y_values.append(e)
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}, avg_past100: {}"
                      .format(e, episodes, time_t, avg_score))
                break

        # Retrain Agent
        if e > 4:
            agent.replay(32)

    # plt.plot(x_values, y_values)
    # plt.show()'


"""
    episode: 1507/2000, score: 199, avg_past100: 197.28
    episode: 1508/2000, score: 199, avg_past100: 197.28
    episode: 1509/2000, score: 199, avg_past100: 197.28
    episode: 1510/2000, score: 199, avg_past100: 197.28
    episode: 1511/2000, score: 199, avg_past100: 197.28
    episode: 1512/2000, score: 199, avg_past100: 197.28
    episode: 1513/2000, score: 199, avg_past100: 197.28
    episode: 1514/2000, score: 199, avg_past100: 197.28
    episode: 1515/2000, score: 199, avg_past100: 197.28
    episode: 1516/2000, score: 199, avg_past100: 197.28
    episode: 1517/2000, score: 199, avg_past100: 197.28 
"""





