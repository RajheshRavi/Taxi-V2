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
    
    state = env.reset()                                                         # Reseting the environment and obtaining States 
    epochs, penality, reward = 0, 0, 0                                          # Initializing Penality and Reward
    done = False
    
    while not done:
        
        env.render()
        
        if r.uniform(0,1) <= epsilon:
            action = env.action_space.sample()                                  # Greedly traversing the states 
        else:
            action = np.argmax(qTable[state])                                   # Choosing the state which has highest reward
        
        nextState,reward,done,info=env.step(action)                             # Performing the chosen action
    
        print("nextState",nextState,"reward",reward,"done",done,"info",info)
    
        oldVal=qTable[state][action]
        nextMax=np.max(qTable[nextState])
    
        newVal=( 1 - alpha ) * oldVal + alpha * ( reward + gamma * nextMax )    
        qTable[state][action]=newVal                                            # Updating the Qtable for the current state
    
        state=nextState                                                         # Updating State
    
        print(reward)
    
        if reward == -10: 
            penality+=1                                                         # Incrementing the penality
        
        epochs+=1                                                              
    
    
    
print(state)
np.save("Qtable")
