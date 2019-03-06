import gym

for e in gym.envs.registry.all():
    print(e)