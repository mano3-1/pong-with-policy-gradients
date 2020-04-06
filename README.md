# pong-with-policy-gradients
AI beats pong game by policy gradients method.

## Requirements:
    Tensorflow 2.0
    keras
    numpy

## policy gradients:
    In policy gradients ,the agent tries to learn policy directly(which is not with the case of value iteration).
    Basically, here we tried to approximate the policy with a neural network.Policy is nothing but it's a 
    probability distrubtion of all possible actions at a state.First we randomly initialize a neural network
    (I used a two layered fully connected network).Then we pick an action and do it and receive the 
    reward accordingly....!!! and we'll collect the rewards of each state until the end of each episode and store them.
    Then we calculate the discounted sum for each  state and use it to train the network.
    
    
## Time taken to converge
    This may take 2-3 days to converge on cpu...one might get the result much faster if one trains 
    on gpu(I haven't tried it but one of my friend said that)
    
## How to run
    Just clone this repository to your system and then run pong.py file.
