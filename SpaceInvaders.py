import gym
from time import sleep

# The action space is all six functions on atari: Up, Down, Left, Right, A, B
# The observation space is an array of rgb pixels: width, height, rgb color
env = gym.make('SpaceInvaders-v0')
print(env.action_space)
print(env.observation_space)



# env.reset()
# for t in range(1000):
#     env.render()
#     action = env.action_space.sample()
#     state, reward, terminal, info = env.step(action)
#     sleep(0.1)



