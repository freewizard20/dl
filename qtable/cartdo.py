# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
import math
import pickle
from time import sleep


## Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v0')

## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

## Defining the simulation related constants
NUM_EPISODES = 1000
MAX_T = 250
STREAK_TO_END = 120
SOLVED_T = 199
DEBUG_MODE = True

q_table = pickle.load(open('qtable.pkl', 'rb'))

def simulate():
    
    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0

    obv = env.reset()
    state_0 = state_to_bucket(obv)

    for episode in range(NUM_EPISODES):        

        env.render()

            # Select an action
        action = select_action(state_0, explore_rate)

            # Execute the action
        obv, reward, done, _ = env.step(action)

            # Observe the result
        state = state_to_bucket(obv)
            
	    # Setting up for the next iteration
        state_0 = state

        sleep(0.1)

    env.close()

def select_action(state, explore_rate):
    action = np.argmax(q_table[state])
    return action

def get_explore_rate(t):
    if t >= 24:
        return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))
    else:
        return 1.0

def get_learning_rate(t):
    if t >= 24:
         return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))
    else:
         return 1.0

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":
    simulate()
