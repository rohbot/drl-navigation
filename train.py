from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
import os
import pickle
import time

mqtt_cmd = "mosquitto_pub -h rohbot.cc -p 1341 -u rohbot -P mqttbr0ker -t drl/"

env = UnityEnvironment(file_name="data/Banana_Linux_NoVis/Banana.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

#print(env, brain_name, brain)

agent = Agent(state_size=37, action_size=4, seed=0)

os.system(mqtt_cmd + "reset -m 0") # reset graph

#agent.qnetwork_local.load_state_dict(torch.load('weights.pth',map_location='cpu'))

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
           
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            
            if done:                                       # exit loop if episode finished
                break
             
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        #if i_episode % 10 == 0:
        print('\rEpisode {}\tScore: {}\tAvg: {:.2f}'.format(i_episode, score, np.mean(scores_window)))
        os.system(mqtt_cmd +  "episode -m " + str(i_episode))
        os.system(mqtt_cmd + "score -m {:.2f}".format(score))
        os.system(mqtt_cmd + "average -m {:.2f}".format(np.mean(scores_window)))
        
        if np.mean(scores_window)>=13:
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            msg = '"Environment solved in {:d} episodes!\tAverage Score: {:.2f}"'.format(i_episode-100, np.mean(scores_window))
            os.system(mqtt_cmd + "done -m " + msg)
            print(msg)
            break
    return scores

scores = dqn()
timestamp =  str(int(time.time()))
torch.save(agent.qnetwork_local.state_dict(), 'cp_'+timestamp+'.pth')
pickle.dump( scores, open( "scores_"+timestamp+".p", "wb" ) )