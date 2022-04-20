import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import collections

# Import Tensorflow libraries

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam


class DQAgent:

    def __init__(self, replayCapacity, inputShape):
        # Initialize replay memory
        self.capacity = replayCapacity
        self.memory = collections.deque(maxlen=self.capacity)
        self.populated = False
        # Policiy model
        self.inputShape = inputShape
        self.policy_model = self.buildNetwork()

        # Target model
        self.target_model = self.buildNetwork()
        self.target_model.set_weights(self.policy_model.get_weights())

    def addToReplayMemory(self, step):
        self.step = step
        self.memory.append(self.step)

    def sampleFromReplayMemory(self, batchSize):
        self.batchSize = batchSize
        if self.batchSize > len(self.memory):
            self.populated = False
            return self.populated
        else:
            return random.sample(self.memory, self.batchSize)

    def buildNetwork(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.inputShape, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dense(24, activation='relu'))
        # model.add(BatchNormalization())
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(
            lr=0.01), metrics=['MeanSquaredError'])
        return model

    def policy_network_fit(self, batch, batchSize):
        self.batchSize = batchSize
        self.batch = batch

    def policy_network_predict(self, state):
        self.state = state
        self.qPolicy = self.policy_model.predict(self.state)
        return self.qPolicy

    def target_network_predict(self, state):
        self.state = state
        self.qTarget = self.target_model.predict(self.state)
        return self.qTarget

    def update_target_network(self):
        self.target_model.set_weights(self.policy_model.get_weights())


# Model parameters
DISCOUNT = 0.90
REPLAY_MEMORY_CAPACITY = 10000
# MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
BATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_INTERVAL = 500
EPSILON = 1  # Exploration percentage
MIN_EPSILON = 0.01
POSSIBLE_ACTIONS = [0, 1, 2]
DECAY = 0.9999


# Create Cartpole environment
env = gym.make('MountainCar-v0')
state = env.reset()
done = False

# create DQN agent

agent = DQAgent(replayCapacity=REPLAY_MEMORY_CAPACITY, inputShape=state.shape)


# Fill the replay memory with the first batch of samples

updateCounter = 0
rewardHistory = []

position_old = 0

for episode in range(200):
    episodeReward = 0
    stepCounter = 0  # count the number of successful steps within the episode

    state = env.reset()
    done = False

    while not done:
        # env.render()

        r = random.random()

        if r <= EPSILON:
            action = random.sample(POSSIBLE_ACTIONS, 1)[0]
            # print('exploration')
        else:
            # print('exploitation')
            qValues = agent.policy_network_predict(state.reshape(1, -1))
            action = np.argmax(qValues[0])

        newState, reward, done, info = env.step(action)

        if (done) and (stepCounter < 199):
            reward += -10

        # Custom rewards
        position = newState[0]
        speed = newState[1]
        if -1 > position > -1.20:
            reward += 100

        if position > 0.25:
            reward += 100

        reward += abs(position)*(abs(speed)*1000)

        if (done) and (position >= 0.5):
            print(
                '----------------------------------REACHED TOP----------------------------------')
            reward = 10000

        stepCounter += 1
        # store step in replay memory
        step = (state, action, reward, newState, done)
        agent.addToReplayMemory(step)
        state = newState
        episodeReward += reward

        # When enough steps in replay memory -> train policy network
        if len(agent.memory) >= (BATCH_SIZE):
            EPSILON = DECAY * EPSILON
            if EPSILON < MIN_EPSILON:
                EPSILON = MIN_EPSILON
            # sample minibatch from replay memory
            miniBatch = agent.sampleFromReplayMemory(BATCH_SIZE)
            miniBatch_states = np.asarray(
                list(zip(*miniBatch))[0], dtype=float)
            miniBatch_actions = np.asarray(list(zip(*miniBatch))[1], dtype=int)
            miniBatch_rewards = np.asarray(
                list(zip(*miniBatch))[2], dtype=float)
            miniBatch_next_state = np.asarray(
                list(zip(*miniBatch))[3], dtype=float)
            miniBatch_done = np.asarray(list(zip(*miniBatch))[4], dtype=bool)

            current_state_q_values = agent.policy_network_predict(
                miniBatch_states)
            y = current_state_q_values

            r2 = random.random()

            if r2 <= EPSILON:
                next_state_q_values = agent.target_network_predict(
                    miniBatch_next_state)

                max_q_next_state = next_state_q_values[:, random.sample(
                    range(len(next_state_q_values[1])), 1)]
            else:
                next_state_q_values = agent.target_network_predict(
                    miniBatch_next_state)
                max_q_next_state = np.max(next_state_q_values, axis=1)

            for i in range(BATCH_SIZE):
                if miniBatch_done[i]:
                    y[i, miniBatch_actions[i]] = miniBatch_rewards[i]
                else:
                    y[i, miniBatch_actions[i]] = miniBatch_rewards[i] + \
                        DISCOUNT * max_q_next_state[i]
            agent.policy_model.fit(miniBatch_states, y,
                                   batch_size=BATCH_SIZE, verbose=0)

        else:
            # env.render()
            continue
        if updateCounter == UPDATE_TARGET_INTERVAL:
            agent.update_target_network()
            print('target updated')
            updateCounter = 0
        updateCounter += 1
    print('episodeReward for episode ', episode, '= ',
          episodeReward, 'with epsilon = ', EPSILON)
    rewardHistory.append(episodeReward)

env.close()

plt.plot(rewardHistory)
plt.show()


#actions = agent.policy_network_predict(state)

#action = np.argmax(actions)
# print(action)

#state, reward, done, info = env.step(action)
# print(reward)
