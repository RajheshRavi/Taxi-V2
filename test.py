import gym
import numpy as np

env=gym.make('Taxi-v2')
env=env.unwrapped

state = env.reset()

epochs, penality, reward = 0, 0, 0
done = False

QTable=np.load("Qtable.npy")    
while not done:
        
    env.render()
    
    action = np.argmax(QTable[state])
        
    nextState,reward,done,info=env.step(action)
    
    print("nextState",nextState,"reward",reward,"done",done,"info",info)
    '''
    oldVal=qTable[state][action]
    nextMax=np.max(qTable[nextState])
    
    newVal=( 1 - alpha ) * oldVal + alpha * ( reward + gamma * nextMax )
    qTable[state][action]=newVal
   ''' 
    state=nextState
    
    #print("Current Reward:",reward)
    
    if reward == -10: 
        penality+=1
        
    epochs+=1
print("Total Penality:",penality)
print("Total Epochs:",epochs)