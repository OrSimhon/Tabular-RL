import numpy as np
import matplotlib.pyplot as plt


class Qlearning_Agent:
    def __init__(self, env, lr, num_episodes, epsilon, decay_factor, discount_rate, max_steps, seed):
        self.env = env
        self.lr = lr
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.discount_rate = discount_rate
        self.max_steps = max_steps
        self.seed = seed
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.pi = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.q = np.zeros((self.n_states, self.n_actions))
        self.set_seed(seed)

    def set_seed(self, SEED):
        np.random.seed(SEED)
        self.env.seed(SEED)
        self.env.action_space.seed(SEED)

    def choose_action(self, state):
        return np.random.choice(self.n_actions, 1, p=self.pi[state]).item()

    def q_update(self, reward, s, a, s_):
        self.q[s][a] += self.lr * (reward + self.discount_rate * self.q[s_, :].max() - self.q[s][a])

    def epsilon_decay(self):
        """
        Exponential decay
        """
        self.epsilon = max(self.epsilon * self.decay_factor, 0.001)

    def epsilon_greedy_policy_from_q(self):
        for state in range(self.n_states):
            self.pi[state, :] = self.epsilon / self.n_actions
            self.pi[state, np.argmax(self.q[state])] = self.epsilon / self.n_actions + 1 - self.epsilon

    @staticmethod
    def plot(avg_rewards, avg_steps):
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(avg_steps)) * 100, avg_rewards)
        plt.title("Mean reward in the last 100 episodes", fontsize=20)
        plt.xlabel('Episode number', fontsize=15)
        plt.ylabel('Reward', fontsize=15)

        plt.figure(figsize=(8, 6))

        plt.plot(np.arange(len(avg_steps)) * 100, avg_steps)
        plt.title("Mean Steps to the goal", fontsize=20)
        plt.xlabel('Episode number', fontsize=15)
        plt.ylabel('Steps', fontsize=15)

        plt.tight_layout()
        plt.show()

    def train(self):
        total_rew_per_ep, av_steps2goal = [], []
        avg_reward, steps = [], []
        step = 0
        for episode in range(self.num_episodes):
            self.epsilon_greedy_policy_from_q()
            state = self.env.reset()
            cumulative_reward_episode = 0
            for step in range(self.max_steps):
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.q_update(reward, state, action, next_state)
                state = next_state
                cumulative_reward_episode += reward
                if done:
                    break

            if cumulative_reward_episode < 1:
                steps.append(self.max_steps)
            else:
                steps.append(step)

            total_rew_per_ep.append(cumulative_reward_episode)
            if (episode + 1) % 100 == 0:  # Every 100 episodes
                av_steps2goal.append(np.mean(steps[-100:]))
                avg_reward.append(np.mean(total_rew_per_ep[-100:]))

            self.epsilon_decay()
        self.plot(avg_reward, av_steps2goal)
        self.env.close()


if __name__ == '__main__':
    import gym

    env = gym.make('FrozenLake-v1', is_slippery=True)
    q_learning_agent = Qlearning_Agent(env=env, lr=0.1, num_episodes=10_000, epsilon=1, decay_factor=0.9995,
                                       discount_rate=0.995,
                                       max_steps=100, seed=1234)
    q_learning_agent.train()
    print(q_learning_agent.pi.argmax(axis=1).reshape(4, 4))
#
# [[0 3 3 3]
#  [0 0 0 0]
#  [3 1 0 0]
#  [0 2 1 0]]
