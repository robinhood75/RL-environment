import numpy as np
from environments import BaseEnvironment


class BaseQLearning:
    def __init__(self, env: BaseEnvironment, gamma, lr=0.1, eps=0.2, max_steps=100):
        self.env = env
        self.gamma = gamma
        assert 0 <= gamma < 1
        self.lr = lr
        self.Q = self.get_q0()
        self.eps = eps
        self.max_steps = max_steps

    def get_q0(self):
        raise NotImplementedError

    def run(self, s0, n_episodes):
        raise NotImplementedError


class QLearning(BaseQLearning):
    def __init__(self, env: BaseEnvironment, gamma, lr=0.1, eps=0.2, max_steps=100):
        super().__init__(env, gamma, lr=lr, eps=eps, max_steps=max_steps)

    def get_q0(self, init='optimistic'):
        if init == 'optimistic':
            factor = 1/(1 - self.gamma)
        elif init == 'zero':
            factor = 0
        else:
            raise ValueError("Unknown initialization type")
        return {s: factor * np.ones(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}

    def _get_action_eps_greedy(self, s):
        s_index = self.env.states_indices[s]
        n_actions = len(self.env.actions[s_index])
        assert n_actions > 0
        d = np.ones(n_actions)
        if n_actions > 1:
            d *= self.eps/(n_actions - 1)
            d[np.argmax(self.Q[s])] = 1 - self.eps
        return np.random.choice(range(n_actions), p=d)

    def run(self, s0, n_episodes):
        for n in range(n_episodes):
            self.env.reset(s0)
            n_steps = 0
            while not self.env.is_terminal(self.env.s) and n_steps < self.max_steps:
                s, s_index = self.env.s, self.env.states_indices[self.env.s]
                a_index = self._get_action_eps_greedy(s)
                r, new_s = self.env.step(action=self.env.actions[s_index][a_index], perform_action=True)
                q_new_s = np.max(self.Q[new_s]) if not self.env.is_terminal(new_s) else 0
                self.Q[s][a_index] += self.lr * (r + self.gamma * q_new_s - self.Q[s][a_index])
                n_steps += 1


class QRM(BaseQLearning):
    """Q-Learning for Reward Machines (Icarte & al., 2018)"""
    def __init__(self, env, gamma, lr=0.1, eps=0.2, max_steps=100):
        super().__init__(env, gamma, lr=lr, eps=eps, max_steps=max_steps)

    def get_q0(self, init='optimistic'):
        if init == 'optimistic':
            factor = 1 / (1 - self.gamma)
        elif init == 'zero':
            factor = 0
        else:
            raise ValueError("Unknown initialization type")
        return {
            u: {s: factor * np.ones(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}
            for u in self.env.rm.states}

    def _get_action_eps_greedy(self, u, s):
        n_actions = len(self.env.actions[self.env.states_indices[s]])
        assert n_actions > 0
        d = np.ones(n_actions)
        if n_actions > 1:
            d *= self.eps/(n_actions - 1)
            d[np.argmax(self.Q[u][s])] = 1 - self.eps
        return np.random.choice(range(n_actions), p=d)

    def run(self, s0, n_episodes):
        for n in range(n_episodes):
            self.env.reset(s0)
            n_steps = 0
            while not self.env.is_terminal(self.env.s) and n_steps < self.max_steps:
                u, s, s_index = self.env.rm.u, self.env.s, self.env.states_indices[self.env.s]
                a_index = self._get_action_eps_greedy(u, s)
                new_s = self.env.get_next_state(self.env.actions[s_index][a_index])
                for v in self.env.rm.states:
                    self.env.rm.u = v
                    r, new_v = self.env.rm.step(new_s, perform_transition=False)
                    q_new = np.max(self.Q[new_v][new_s]) if not self.env.is_terminal(new_s) else 0
                    self.Q[v][s][a_index] += self.lr * (r + self.gamma * q_new - self.Q[v][s][a_index])
                self.env.s = new_s
                n_steps += 1


class OptimisticQLearning(BaseQLearning):
    pass

