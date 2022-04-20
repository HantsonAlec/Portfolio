# Implementation of REINFORCE for the Cartpole environment
import gym
import numpy as np

import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.get_logger().setLevel('ERROR')


class REINFORCE:
    def __init__(self, env, path=None):
        self.env = env
        self.state_shape = env.observation_space.shape  # the state space
        self.action_shape = env.action_space.n  # the action space
        self.gamma = 0.99  # decay rate of past observations
        self.alpha = 1e-4  # learning rate of gradient
        self.learning_rate = 0.01  # learning of deep learning model
        # self.alpha_decay = 0.9995
        # self.min_alpha = 0.01

        if not path:
            self.model = self.build_policy_network()  # build model
        else:
            self.model = self.load_model(path)  # import model

        # record observations
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        self.total_rewards = []

    def build_policy_network(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=self.state_shape))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=self.learning_rate), metrics=['accuracy'])
        return model

    def hot_encode_action(self, action):

        action_encoded = np.zeros(self.action_shape)
        action_encoded[action] = 1

        return action_encoded

    def remember(self, state, action, action_prob, reward):
        encoded = self.hot_encode_action(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.gradients.append(encoded-action_prob)
        self.probs.append(action_prob)
        # STORE EACH STATE, ACTION AND REWARD into the episodic momory #############################

    def compute_action(self, state):
        state = state.reshape([1, state.shape[0]])
        action_probability_distribution = self.model.predict(state).flatten()
        action_probability_distribution /= np.sum(
            action_probability_distribution)
        action = np.random.choice(
            self.action_shape, 1, p=action_probability_distribution)[0]

        # COMPUTE THE ACTION FROM THE SOFTMAX PROBABILITIES
        # if self.alpha>self.min_alpha:
        #   self.alpha = self.alpha*self.alpha_decay
        return action, action_probability_distribution

    def get_discounted_rewards(self, rewards):

        discounted_rewards = []
        cumulative_total_return = 0
        # iterate the rewards backwards and and calc the total return
        for reward in rewards[::-1]:
            cumulative_total_return = (
                cumulative_total_return*self.gamma)+reward
            discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards -
                                   mean_rewards)/(std_rewards+1e-7)  # avoiding zero div
        return norm_discounted_rewards

    def train_policy_network(self):
        # get X_train
        states = np.vstack(self.states)

        # get y_train
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        discounted_rewards = self.get_discounted_rewards(rewards)
        gradients = (gradients*discounted_rewards)
        gradients = self.alpha*np.vstack([gradients])+self.probs
        y_train = gradients
        history = self.model.train_on_batch(states, y_train)

        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

        return history

    def train(self, episodes):
        rewardHistory = []
        for episode in range(episodes):
            episodeReward = 0
            max_t = 200
            state = self.env.reset()
            done = False
            for t in range(max_t):
                action, action_prob = self.compute_action(state)
                newState, reward, done, info = self.env.step(action)
                self.remember(state, action, action_prob, reward)
                state = newState
                episodeReward += reward
                if done:
                    break
            self.train_policy_network()
            print('episodeReward for episode ', episode, 'reward = ',
                  episodeReward, 'with alpha = ', self.alpha)
            rewardHistory.append(episodeReward)
        return rewardHistory

    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_shape)
        action_encoded[action] = 1

        return action_encoded


ENV = "CartPole-v1"

N_EPISODES = 500


def running_average(data, N):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# set the env
env = gym.make(ENV)  # env to import
env.reset()  # reset to env

Agent = REINFORCE(env)

episodic_reward_history = Agent.train(N_EPISODES)

MA10 = running_average(episodic_reward_history, 100)

plt.plot(np.arange(len(episodic_reward_history)),
         episodic_reward_history, MA10)
plt.show()
