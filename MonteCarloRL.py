import numpy as np
import matplotlib.pyplot as plt


class MCAgent:
    def __init__(self, env, lr, num_episodes, epsilon, decay_factor, discount_rate, seed):
        self.env = env
        self.lr = lr
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.discount_rate = discount_rate
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

    def generate_episode(self):
        states, actions, rewards = [], [], []
        state = self.env.reset()
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        return np.array(states), np.array(actions), rewards

    def calc_returns(self, rewards):
        returns = np.zeros(len(rewards))
        gt = 0
        for t in range(1, len(returns) + 1):
            gt = rewards[-t] + self.discount_rate * gt
            returns[-t] = gt

        return returns

    def q_update(self, states, actions, returns):
        """
        First visit
        """
        visited = np.zeros((self.n_states, self.n_actions))
        for t, (state, action) in enumerate(zip(states, actions)):
            if visited[state][action] == 0:
                self.q[state][action] += self.lr * (returns[t] - self.q[state][action])
                visited[state][action] = 1

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
    def plot(avg_rewards):
        plt.figure(figsize=(8, 6))
        plt.plot(avg_rewards)
        plt.title("Mean reward in the last 100 episodes", fontsize=20)
        plt.xlabel('Episode number', fontsize=15)
        plt.ylabel('Reward', fontsize=15)
        plt.tight_layout()
        plt.show()

    def train(self):
        total_rew_per_ep = []
        avg_reward = []
        for episode in range(self.num_episodes):
            self.epsilon_greedy_policy_from_q()
            states, actions, rewards = self.generate_episode()
            total_rew_per_ep.append(sum(rewards))
            if sum(rewards) != 0 or self.epsilon < 1:
                returns = self.calc_returns(rewards)
                self.q_update(states, actions, returns)
                self.epsilon_decay()
            if (episode + 1) % 100 == 0:  # Every 100 episodes
                avg_reward.append(np.mean(total_rew_per_ep[-100:]))
        self.plot(avg_reward)
        self.env.close()


if __name__ == '__main__':
    import gym

    env = gym.make('FrozenLake-v1', is_slippery=True)
    mc_agent = MCAgent(env=env, lr=0.01, num_episodes=20_000, epsilon=1, decay_factor=0.9995, discount_rate=0.995,
                       seed=1234)
    mc_agent.train()

    print(mc_agent.pi.argmax(axis=1).reshape(4, 4))

# [[0 3 3 3]
#  [0 0 0 0]
#  [3 1 2 0]
#  [0 2 1 0]]
