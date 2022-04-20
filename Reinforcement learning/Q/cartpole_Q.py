import gymgrid
import numpy as np
import gym
import random
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')


class Qagent:
    def __init__(self, nr_actions, alpha, lr_decay, gamma, epsilon, decay):
        print('exploration rate', epsilon)
        self.alpha = alpha
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.nr_actions = nr_actions
        self.state = 0
        self.possibleActions = [i for i in range(nr_actions)]
        self.nr_bins = [30, 30, 50, 50]
        self._state_bins = [
            # Cart position.
            np.linspace(-2.4, 2.4, self.nr_bins[0] + 1)[1:-1],
            # Cart velocity.
            np.linspace(-3.0, 3.0, self.nr_bins[1] + 1)[1:-1],
            # Pole angle.
            np.linspace(-.5, .5, self.nr_bins[2] + 1)[1:-1],
            # Tip velocity.
            np.linspace(-2.0, 2.0, self.nr_bins[3] + 1)[1:-1],
        ]

        nr_states = max(self.nr_bins)**len(env.observation_space.high)
        self.q_table = np.zeros((nr_states, nr_actions))

    @staticmethod
    def _discretize_value(value, bins):
        return np.digitize(x=value, bins=bins)

    def _build_state(self, observation):
        # Discretize the observation features and reduce them to a single integer.
        state = sum(
            self._discretize_value(
                feature, self._state_bins[i]) * ((self.nr_bins[i]) ** i)
            for i, feature in enumerate(observation)
        )
        return state

    def compute_action(self, state):
        # self.state = state
        self.state = self._build_state(state)
        if self.epsilon > 0.05:
            self.epsilon = self.epsilon*self.decay
        if self.alpha > 0.01:
            self.alpha = self.alpha*self.lr_decay
        self.r = np.random.uniform()
        if self.r <= self.epsilon:
            self.action = random.sample(self.possibleActions, 1)[0]
            # print('\nexploration')
        else:
            # print('\nexploitation')
            self.action = np.argmax(self.q_table[self.state, :])

        return self.action

    def update_qtable(self, state, new_state, reward):
        self.reward = reward
        # self.state = state
        self.state = self._build_state(state)
        # self.new_state = new_state
        self.new_state = self._build_state(new_state)
        self.q_table[self.state, self.action] = (1 - self.alpha) * self.q_table[
            self.state, self.action] + self.alpha * (self.reward + self.gamma * self.q_table[
                self.new_state, np.argmax(self.q_table[self.new_state, :])])


def running_average(data, N):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


nr_of_episodes = 60000
alpha = 1
lr_decay = 0.99995
gamma = 0.95
decay = 0.99995
epsilon = 1
nr_actions = 2

episodic_reward_history = []
max_t = 200
q_agent = Qagent(nr_actions, alpha, lr_decay, gamma, epsilon, decay)
for i_episode in range(nr_of_episodes):
    # env.render()
    state = env.reset()
    episodic_reward = 0
    done = False
    for t in range(max_t):
        # compute and execute action
        action = q_agent.compute_action(state)
        new_state, reward, done, dict = env.step(action)
        episodic_reward += reward

        # update the q-table
        # print(env.step(action))

        # penalty for ending to early
        if done and t < max_t - 1:
            reward = -max_t
        elif done and t >= max_t-1:
            reward += 300
        q_agent.update_qtable(state, new_state, reward)

        # update the state
        state = new_state

        # # env.render()
        # print('\n', new_state)

        if done == True:
            break
    print(f"Episode: {i_episode}, reward: {episodic_reward}")
    episodic_reward_history.append(episodic_reward)


print(q_agent.q_table)

# Test final q-table
done = False
state = env.reset()
env.render()
stopcounter = 0
while (done == False):
    q_agent.epsilon = 0
    action = q_agent.compute_action(state)
    new_state, reward, done, dict = env.step(action)
    state = new_state
    stopcounter += 1
    if stopcounter > 25:
        done = True
    env.render()
    print('\n', state)


print(episodic_reward_history)
MA10 = running_average(episodic_reward_history, 100)

plt.plot(np.arange(len(episodic_reward_history)),
         episodic_reward_history, MA10)
plt.show()
