# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:49:51 2020

@author: Lenovo
"""

from keras.layers import Dense,Activation,Input
from keras.models import Model,load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

class Agent(object):
    def __init__(self,ALPHA,GAMMA=0.99,n_actions=4,layer1_size=16,
                 layer2_size=16,input_dims=128,name='reinforce.h5'):
        self.lr = ALPHA
        self.gamma = GAMMA
        self.G = 0
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory=[]
        self.action_memory=[]
        self.reward_memory=[]
        
        self.policy,self.predict=self.build_policy_net()
        self.action_space=[i for i in range(n_actions)]
        self.model_file=name
    def build_policy_net(self):
        input=Input(shape=(self.input_dims,))
        advantages=Input(shape=[1])
        dense1=Dense(self.fc1_dims,activation='elu')(input)
        dense2=Dense(self.fc2_dims,activation='elu')(dense1)
        probs=Dense(self.n_actions,activation='softmax')(dense2)
        
        def loss(y_true,y_pred):
            out=K.clip(y_pred,1e-8,1-1e-8)
            log_like=y_true*K.log(out)
            
            return K.sum(-log_like*advantages)
        
        policy=Model(inputs=[input,advantages],output=[probs])
        policy.compile(optimizer=Adam(lr=self.lr),loss=loss)
        
        predict = Model(input=[input],output=[probs])
        
        
        return policy,predict
    
    
    
    def choose_action(self,observation):
        state=observation[np.newaxis,:]
        probabilities=self.predict.predict(state)[0]
        action=np.random.choice([2,3],p=probabilities)
        
        return action
    
    
    def store_transition(self,observation,action,reward):
        
        self.state_memory.append(observation)
        self.reward_memory.append(reward)
        self.action_memory.append(action)
        
    def learn(self):
        
        state_memory=np.array(self.state_memory)
        action_memory=np.array(self.action_memory)
        reward_memory=np.array(self.reward_memory)
        
        actions=np.zeros([len(action_memory),self.n_actions])
        for i in range(len(list(action_memory))):
            if action_memory[i]==2:
                actions[i][0]=1
                actions[i][1]=0
            else:
                actions[i][0]=0
                actions[i][1]=1                
           
        G=np.zeros_like(reward_memory)
        
        for t in range(len(reward_memory)):
            G_sum=0
            discount=1
            for k in range(t,len(reward_memory)):
                G_sum+=reward_memory[k]*discount
                discount*=self.gamma
                
            G[t]=G_sum
        mean=np.mean(G)
        std=np.std(G) if np.std(G)>0 else 1
        self.G=(G-mean)/std
        
        
        cost=self.policy.train_on_batch([state_memory,self.G],actions)
        
        self.action_memory=[]
        self.reward_memory=[]
        self.state_memory=[]
        
    def save_model(self):
        self.policy.save(self.model_file)
        
        
    def load_model(self):
        self.policy=load_model(self.model_file)
        
