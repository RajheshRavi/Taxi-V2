import gym
import numpy as np
import random as r

env = gym.make('Taxi-v2')
env=env.unwrapped



alpha=0.1
gamma=0.6
epsilon=0.1


qTable=np.zeros([env.observation_space.n,env.action_space.n])

for i in range(10000):
    
    state = env.reset()
    epochs, penality, reward = 0, 0, 0
    done = False
    
    while not done:
        
        env.render()
        
        if r.uniform(0,1) <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qTable[state])
        
        nextState,reward,done,info=env.step(action)
    
        print("nextState",nextState,"reward",reward,"done",done,"info",info)
    
        oldVal=qTable[state][action]
        nextMax=np.max(qTable[nextState])
    
        newVal=( 1 - alpha ) * oldVal + alpha * ( reward + gamma * nextMax )
        qTable[state][action]=newVal
    
        state=nextState
    
        print(reward)
    
        if reward == -10: 
            penality+=1
        
        epochs+=1
    
    
    
print(state)
np.save("Qtable")