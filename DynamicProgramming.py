import numpy as np


class DPAgent:
    def __init__(self, env, discount_rate, seed, theta):
        self.env = env
        self.discount_rate = discount_rate
        self.theta = theta
        self.set_seed(seed)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.v = np.zeros(self.n_states)
        self.pi = np.ones((self.n_states, self.n_actions)) / self.n_actions

    def set_seed(self, SEED):
        np.random.seed(SEED)
        self.env.seed(SEED)
        self.env.action_space.seed(SEED)

    def bellman_optimality_update(self, s):
        q_s = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            for prob, next_state, reward, done in self.env.P[s][a]:
                q_s[a] += prob * reward + self.discount_rate * prob * self.v[next_state]
        self.v[s] = np.max(q_s)

    def bellman_expectation(self):
        v = np.zeros(self.n_states)
        delta = self.theta + 1
        while delta > self.theta:
            delta = 0
            for s in range(self.n_states):
                v_temp = v[s]
                q_s = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        q_s[a] += prob * reward + self.discount_rate * prob * v[next_state]
                v[s] = (self.pi[s] * q_s).sum()
                delta = max(delta, abs(v_temp - v[s]))
        self.v = np.copy(v)

    def greedy_policy_from_value(self):
        self.pi = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            q_s = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    q_s[a] += prob * reward + self.discount_rate * prob * self.v[next_state]
            self.pi[s][np.argmax(q_s)] = 1

    def value_iteration(self):
        delta = self.theta + 1
        while delta > self.theta:
            delta = 0
            for s in range(self.n_states):
                v = self.v[s]
                self.bellman_optimality_update(s)
                delta = max(delta, abs(v - self.v[s]))
        self.greedy_policy_from_value()
        self.env.close()

    def policy_iteration(self):
        policy_stable = False
        while not policy_stable:
            pi = np.copy(self.pi)
            self.bellman_expectation()  # Policy Evaluation
            self.greedy_policy_from_value()  # Policy Improvement
            if (pi == self.pi).all():
                policy_stable = True
        self.env.close()

    def policy_from_one_hot(self):
        p = np.zeros(16)
        for i, s in enumerate(self.pi):
            p[i] = s.argmax()
        return p.reshape(self.n_actions, self.n_actions)


def print_final(agent, value_iteration):
    print(f"{'Value' if value_iteration else 'Policy'} Iteration\n{'-' * 20}")
    print(f"\nOptimal Value Function:\n{agent.v.reshape(4, 4)}\n")
    print(f"\nOptimal Policy:\n{agent.policy_from_one_hot()}\n")


if __name__ == '__main__':
    import gym

    env = gym.make('FrozenLake-v1', is_slippery=False)

    vi_agent = DPAgent(env, discount_rate=0.99, seed=1234, theta=1e-8)
    vi_agent.value_iteration()
    print_final(vi_agent, value_iteration=True)

    pi_agent = DPAgent(env, discount_rate=0.99, seed=1234, theta=1e-8)
    pi_agent.policy_iteration()
    print_final(pi_agent, value_iteration=False)
#
# [[0. 3. 3. 3.]
#  [0. 0. 0. 0.]
#  [3. 1. 0. 0.]
#  [0. 2. 1. 0.]]
