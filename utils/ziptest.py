import numpy as np
import gym

a = [1,2,3,4,5]
b = ['a','b','c','d','e']

for x,y in zip(a,b):
    print(x,y)

print(zip(a,b))

#for index, i in a:
#    print(i)

for x in zip(a,b):
    print(x)

print("test: ",(1,2,3,4)+(5,))

print("from here")

env = gym.make('CartPole-v0')

test = (1,1,6,3)
test2 = 2;
q_table = np.zeros(test + (test2,))
print(q_table)
print(q_table[(0,0,1,1)])
print(np.amax(q_table[(0,0,1,1)]))
print('s : ' , q_table[(0,0,1,1,1,)])
print('z : ', q_table[0][0][1])
print('w : ', q_table[(0,0,1)])
print(env.observation_space.low)
print(env.observation_space.high)
print(list(zip(env.observation_space.low, env.observation_space.high)))
