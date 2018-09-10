import gym
import numpy as np

env = gym.make('CartPole-v0')
print(env.action_space.n)
print((env.action_space.n,))
print(env.observation_space)

sam = np.zeros(10)
sam[2] = 12
test = np.argmax(sam)
print(sam)
print(test)
env.close()
