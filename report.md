# Udacity Exercise Navigation Report
The navigation project is using Deep Q-learing to train an agent to collect yellow bananas in an unity environment.
## The Environment Description
In this project, I have trained an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
## Learning Algorithm
I have used deep Q-learning algorithm to train the agent. Please read this [research paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) to get more insights about the topic

### The Q-Network Architecture 

    QNetwork(
      (Fully connected layer1): Linear(in_features=37, out_features=64, bias=True)
      (Fully connected layer2): Linear(in_features=64, out_features=64, bias=True)
      (Fully connected layer3): Linear(in_features=64, out_features=64, bias=True)
      (Fully connected layer4): Linear(in_features=64, out_features=64, bias=True)
      (Fully connected layer5): Linear(in_features=64, out_features=4, bias=True)
    ) 
### hyper-parameter

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size
    GAMMA = 0.95  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 5  # how often to update the network
    eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    eps_end (float): minimum value of epsilon
    eps_decay (float): multiplicative factor (per episode) for decreasing epsilon

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    
### some advice to choose hyper-parameter
#### eps
0<= ε <=1<br>
the agent selects the greedy action with probability 1- ε <br>
the agent selects an action uniform at random from the set of available(non-greedy AND greedy) actions with probability  ε <br>
so if  ε = 0, the agent always select the greedy action<br>
#### GAMMA
0<= 𝞬 <=1<br>
the agent will only care about if immediate rewards if 𝞬 = 0<br>
the agent will care more about future rewards if 𝞬 is large<br>
if you want to care more about future, you should take a large 𝞬

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
![train_network](https://github.com/sand47/Udacity-drlnd-navigation/blob/master/output_graph.PNG)
## More Detail Info
Please see: [Navigation.ipynb](https://github.com/sand47/Udacity-drlnd-navigation/blob/master/Navigation.ipynb)
## Ideas for Future Work
- choose a different network architecture
    - Use conv-net with image pixels as input
    - deeper network 
    - use dropout, bathnorm and use below algorithm to boost DQN
- use different learning algorithm 
    - [prioritized experience replay](https://arxiv.org/abs/1511.05952)
    - [Dueling DQN](https://arxiv.org/abs/1511.06581)
    - [Rainbow](https://arxiv.org/abs/1710.02298)
- tuned the values of the hyper-parameters
    
    
    