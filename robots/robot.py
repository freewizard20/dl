import gym
import time

env = gym.make('Humanoid-v2')
#observation = env.reset()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        time.sleep(1)
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break;

