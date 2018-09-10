import gym
import time

env = gym.make('BipedalWalker-v2')

def selectaction(obv):
    if obv[3]>0:
        return 1;
    else:
        return 0;

def simulate():
    observation = env.reset()

    for t in range(1000):
        time.sleep(0.1)
        env.render()
        observation, reward, done, info = env.step(selectaction(observation))
        print(observation, reward, done, info)

    time.sleep(10000)
    env.close()

if __name__ == "__main__":
    simulate()
