# gymnasium-3202
## CSPB 3202 Final Project - Deep Learning with Gymnasium

## Lunar Lander Deep Q Network

This is an implementation of a Deep Q Network agent in the OpenAI Gymnasium Box2D Lunar Lander environment.

### Environment Description:

From Gymnasium: "This environment is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off."

The environment contains a lunar lander which must be landed between a pair of flags on the moon surface. The lander enters from the center top of the screen and is controlled by thrusters on the bottom, left, and right. It has two legs. In each state you may fire one of the thrusters. The environment can either be discrete or continuous; this implementation uses the discrete version.

------

### Environment Details:

__Action Space:__  Discrete(4)

__Actions:__ 
* 0: do nothing
* 1: fire left engine
* 2: fire bottom engine
* 3: fire right engine

__Observation Space:__ 8-dimensional vector containing the x and y coordinates of the lander, the linear velocity of the lander, the angle and angular velocity of the lander, and two booleans representing whether each leg is in contact with the ground or not

__Rewards:__ The total reward is a sum of the reward at each step, which depends on several factors like how close the lander is to the landing pad, its speed, angle, whether an engine is firing, and whether the legs are in contact with the ground. There is an additional reward at the end of the episode of -100 if the lander crashes or +100 if it lands safely. An episode is a solution if it scores at least 200 points.

__Version:__ "LunarLander-v2"

-------

### Algorithm: Deep-Q Network

A Deep-Q Network (DQN) is a reinforcement learning algorithm that uses Q-learning and deep neural networks. The DQN uses a neural network to approximate the Q-function and trains using a replay buffer, which enables it to use batches of past experiences for training rather than a single experience. This avoids over-fitting. To control the exploration and exploitation rate, a parameter epsilon is used which is gradually reduced.

------

### Process

#### Setup:

Before training, the following classes are defined:
* ReplayMemory - a replay buffer to store previous transitions, used to sample for training. Contains methods to push transitions to a deque, sample from the deque, and return the length of the deque.
* DQN - The DQN policy network. Computes the Q-values of a given state using hidden layers.
* QAgent - the agent to be used for training the DQN and returning actions once trained. Initializes various parameters for training as well as a ReplayMemory buffer and two DQN objects to use in training. Also defines functions for selecting actions and training the DQN.

In addition, a named tuple "Transition" is defined to store information about each transition.

#### Training:

The train function takes an environment and an agent as arguments, as well as parameters for the number of training episodes and epsilon values. For each episode, the function resets the environment, then asks the agent to select an action for each state according to the current DQN policy until the episode reaches a terminal state. Each transition is stored in the replay memory buffer, and the model is optimized after a set number of cycles (defined by the learn_step parameter) using a random sample of transitions from the buffer. After each training episode, epsilon is updated to reduce the frequency of random actions as the policy is honed. A helper function also displays and updates a graph showing the rewards from each episode.

------

### Tools

Gymnasium: https://gymnasium.farama.org/

Pytorch: https://pytorch.org/

Matplotlib: https://matplotlib.org/

Numpy: https://numpy.org/

IPython: https://ipython.org/




