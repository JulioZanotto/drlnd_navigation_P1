from dqn_agent import Agent
from collections import deque
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment
import numpy as np
import argparse


def evaluate(path='./Banana_Linux/Banana.x86_64', load_model='chk_banana.pth'):

    env = UnityEnvironment(file_name=path)

    # get the default brain
    brain_name = env.brain_names[0]

    agent = Agent(state_size=37, action_size=4, seed=0)
    agent.load(load_model)


    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state, 0.)                 # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(score))

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Udacity DQN training agent')
    parser.add_argument('--path', action='store', dest='path',
                        default='./Banana_Linux/Banana.x86_64', 
                        help='Path to the unpacked banana environment')
    parser.add_argument('--model', action='store', dest='model',
                        default='chk_banana.pth', 
                        help='Path to the saveed pth model')
    arguments = parser.parse_args()
    evaluate(path=arguments.path, load_model=arguments.model)