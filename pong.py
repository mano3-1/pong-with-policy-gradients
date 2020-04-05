# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 00:48:56 2020

@author: Lenovo
"""

import numpy as np

from policy_iteration import Agent
#from utils import plotLearning
import gym

if __name__=='__main__':
    agent=Agent(ALPHA=0.0005,input_dims=6400,GAMMA=0.99
                ,n_actions=2,layer1_size=64,layer2_size=64)
    env=gym.make('Pong-v0')
    score_history=[]
    def preprocess(image):
        image=image[35:195]
        image=image[::2,::2,0]
        image[image==144]=0
        image[image==109]=0
        image[image!=0]=1
        return image.astype('float').ravel()
        
    n_episodes=2000
    
    for i in range(n_episodes):
        done=False
        score=0
        obs=env.reset()
        prev_s=0
        while not done:
            env.render()
            cur_s=preprocess(obs)
            observation=cur_s-prev_s
            action=agent.choose_action(observation)
            obs,reward,done,info=env.step(action)
            agent.store_transition(observation,action,reward)
            prev_s=cur_s
            score+=reward
        score_history.append(score)
        agent.learn()
        
        print('episode',i,'score %.1f'%score,
              'average_Score %.1f'%np.mean(score_history[-100:]))
        #filename='lunar_lander.png'
        #plotLearning(score_history,filename=filename,window=100)
        
        
