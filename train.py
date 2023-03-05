from dqn_agent import Agent
from collections import deque
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment
import numpy as np
import argparse


def train(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, 
          path='./Banana_Linux/Banana.x86_64'):
    
    '''
    Function to train the agent for 2000 episodes.

    Args:
    Path = Path to the Unity Banana environment

    Returns:
    None

    Saves locally the agent as chk_banana_term.pth
    '''

    env = UnityEnvironment(file_name=path)

    # get the default brain
    brain_name = env.brain_names[0]

    agent = Agent(state_size=37, action_size=4, seed=0)

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]   
        score = 0
        done = False
        while not done:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0] 
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=15.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'chk_banana_term.pth')
            break

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Udacity DQN training agent')
    parser.add_argument('--path', action='store', dest='path',
                        default='./Banana_Linux/Banana.x86_64', 
                        help='Path to the unpacked banana environment')
    arguments = parser.parse_args()
    train(path=arguments.path)