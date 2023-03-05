[//]: # (Image References)

[image1]: imgs/plots.png "Plotted Scores"

# Udacity Project One Navigation
The navigation project is using Deep Q-learing to train an agent to collect bananas in an unity environment.

## The Environment Description
In this project, an agent was trained to navigate in a square world to collect bananas.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Learning Algorithm
I have used deep Q-learning algorithm to train the agent. This algorithm is based on Q-learning algorithm, but for continuous state spaces, and for that, as a function approximation, was used a neural network, where the weights have the task to map the continuous state spaces to discrete actions as mentioned above. As a dataset used to model to train, was used a replay buffer to collect experiences, as state, reward, done and next_state. In the learning process, the model samples from the buffer and updates the weights of the neural network. In order to be able to find the loss on what the network predicted and what was considered a reward for that state, a target network is used with the same parameters of the local network, but it is updated in a inferior frequency, and this value is mentioned on the hyper parameters section, as UPDATE_EVERY.

### The Q-Network Architecture 

    QNetwork(
      Input size of 37 (states)
      Fully Connected layer: input_size = 37, output_size = 128 (Activation ReLU)
      Fully Connected layer: input_size = 128, output_size = 64 (Activation ReLU)
      Fully Connected layer: input_size = 64, output_size = 4 (Outputs the q-value for each action)
    )

### hyper-parameter

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network

    n_episodes=2000         # Number of episodes
    eps_start=1.0           # Starting epsilon
    eps_end=0.01            # Final epsilon value
    eps_decay=0.995         # Decaying rate of epsilon

    The value of *GAMMA*, is responsible for the discounting the rewards for the episodes, so the nearest ones have more say in the final reward
    The value of *epsilon* is the rate where you choose the random actions to be chosen for the episodes. Here is the exploration-exploitation dilemma.
    

## Train The Network
    Episode 100	Average Score: 0.76
    Episode 200	Average Score: 4.10
    Episode 300	Average Score: 8.05
    Episode 400	Average Score: 10.59
    Episode 500	Average Score: 12.79
    Episode 600	Average Score: 13.37
    Episode 700	Average Score: 13.87
    Episode 714	Average Score: 14.03
    Environment solved in 614 episodes!	Average Score: 14.03

## Plotted Results

![Plotted Scores][image1]

## More Detail Info
Please see: [Navigation.ipynb](https://github.com/sand47/Udacity-drlnd-navigation/blob/master/Navigation.ipynb)

## Ideas for Future Work
The Agent had a very nice performance, and was able to solve the environment in few steps, less than a 1.000 !
Even thought, a few tweaks can be done, as change a little hyper parameters like the learning rate or the epsilon decay. And also some different architectures like Dueling DQN and prioritized experience replay, which choses from the replay buffer experiences that adds more value! More about them below.

- [prioritized experience replay](https://arxiv.org/abs/1511.05952)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)

    
    
    