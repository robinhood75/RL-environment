import numpy as np
from environments import BaseEnvironment, get_cross_product
from copy import copy
from reward_machines import BaseRewardMachine


class BaseQLearning:
    def __init__(self, env: BaseEnvironment, gamma, lr=0.1, eps=0.2, max_steps=100):
        self.env = env
        self.gamma = gamma
        assert 0 <= gamma <= 1
        self.lr = lr
        self.eps = eps
        self.max_steps = max_steps
        self.Q = self.get_q0()

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
                    r, new_v = self.env.rm.step(s, new_s, perform_transition=False)
                    q_new = np.max(self.Q[new_v][new_s]) if not self.env.is_terminal(new_s) else 0
                    self.Q[v][s][a_index] += self.lr * (r + self.gamma * q_new - self.Q[v][s][a_index])
                self.env.s = new_s
                n_steps += 1


class OptimisticQLearning:
    """The only existing average-reward model-free RL algorithm with regret guarantees (Wei et al. 2019)"""
    def __init__(self, env: BaseEnvironment, t, c=None, h=None, delta=0.1):
        self.env = env
        self.h = self.get_h(h, t, n_states=env.n_states, n_actions=2, delta=delta)  # remove hardcoded n_actions
        self.gamma = 1 - 1/self.h
        self.t = t
        self.c = self.get_c(c, t, n_states=env.n_states, delta=delta)
        print(f"h = {self.h}, c = {self.c}")
        self.Q = self.get_q0()
        self.Q_est = self.get_q0()
        self.V = self.get_v0()
        self.n = {s: np.zeros(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}

    @staticmethod
    def get_c(c, t, n_states, delta=0.1):
        return 4 * n_states * np.sqrt(np.log(2 * t / delta)) if c is None else c

    @staticmethod
    def get_h(h, t, n_states, n_actions, delta=0.1):
        return int((t/(n_actions * n_states * np.log(4 * t / delta)))**(1/3)) + 1 if h is None else h

    def get_v0(self):
        return {s: self.h for s in self.env.states}

    def get_q0(self):
        return {s: self.h * np.ones(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}

    def _get_action(self, s):
        assert self.env.actions[self.env.states_indices[s]] != []
        return np.argmax(self.Q_est[s])

    def run(self, s0):
        s = s0
        for n in range(self.t):
            if self.env.is_terminal(s):
                # Get around weak communication assumption by transitioning to s0
                self.env.reset(s0, reset_rewards=False)
            s = self.env.s

            # Execute action, observe reward & new state
            a_index = self._get_action(s)
            r, new_s = self.env.step(self.env.actions[self.env.states_indices[s]][a_index])

            # Update variables
            self.n[s][a_index] += 1
            tau = self.n[s][a_index]
            lr = (self.h + 1) / (self.h + tau)
            bias = self.c * np.sqrt(self.h / tau)

            # Learning step
            self.Q[s][a_index] += lr * (r + self.gamma * self.V[new_s] + bias - self.Q[s][a_index])
            self.Q_est[s][a_index] = np.min([self.Q_est[s][a_index], self.Q[s][a_index]])
            self.V[s] = np.max(self.Q_est[s])


class OQLRM(OptimisticQLearning):
    """Optimistic Q-Learning for reward machines"""
    def __init__(self, env: BaseEnvironment, t, c=None, delta=0.1):
        super().__init__(env=env, t=t, c=c, delta=delta)

    def get_q0(self):
        return {u: {s: self.h * np.ones(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}
                for u in self.env.rm.states}

    def get_v0(self):
        return {u: {s: self.h for s in self.env.states} for u in self.env.rm.states}

    def _get_action(self, s):
        assert self.env.actions[self.env.states_indices[s]] != []
        return np.argmax(self.Q_est[self.env.rm.u][self.env.s])

    def run(self, s0):
        s = s0
        for n in range(self.t):
            if self.env.is_terminal(s):
                # Get around weak communication assumption by transitioning to s0
                self.env.reset(s0, reset_rewards=False)
            s = self.env.s

            # Take action and update parameters
            a_index = self._get_action(s)
            action = self.env.actions[self.env.states_indices[s]][a_index]
            _, new_s = self.env.step(action=action, perform_action=True)
            next_u = self.env.rm.u

            self.n[s][a_index] += 1
            tau = self.n[s][a_index]
            lr = (self.h + 1) / (self.h + tau)
            bias = self.c * np.sqrt(self.h / tau)

            for u in self.env.rm.states:
                self.env.rm.u = u
                r, new_u = self.env.rm.step(s, new_s, perform_transition=False)
                self.Q[u][s][a_index] += lr * (r + self.gamma * self.V[u][new_s] + bias - self.Q[u][s][a_index])
                self.Q_est[u][s][a_index] = np.min([self.Q_est[u][s][a_index], self.Q[u][s][a_index]])
                self.V[u][s] = np.max(self.Q_est[u][s])

            self.env.rm.u = next_u


class UCBQL(BaseQLearning):
    def __init__(self, env: BaseEnvironment, lr=0.1, eps=0.2, max_steps=100, bonus="hoeffding", c=2, delta=0.05,
                 random_reset=True, iota_type=3, c1=0.5, c2=0.5, reward_shaping=None):
        super().__init__(env=env, gamma=1, lr=lr, eps=eps, max_steps=max_steps)
        assert bonus in ["hoeffding", "bernstein"]
        self.random_reset = random_reset
        self.bonus = bonus
        self.n = {h: {s: np.zeros(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}
                  for h in range(self.max_steps)}
        self.c = c
        self.V = self.get_v0()
        self.iota_type = iota_type
        self.delta = delta
        self.mu = self.get_mu0()
        self.sigma = self.get_mu0()
        self.beta = self.get_mu0()
        self.c1 = c1
        self.c2 = c2
        if self.bonus == "bernstein":
            assert c1 is not None and c2 is not None

    def get_q0(self):
        return {h:
                {s: self.max_steps * np.ones(len(self.env.actions[self.env.states_indices[s]]))
                 for s in self.env.states}
                for h in range(self.max_steps)}

    def get_v0(self):
        return {h: {s: 0 for s in self.env.states} for h in range(self.max_steps)}

    def _get_action(self, h, s):
        assert self.env.actions[self.env.states_indices[s]] != []
        return np.argmax(self.Q[h][s])

    def run(self, s0, n_episodes):
        # TODO: change "n_episodes"'s name
        if self.iota_type in [1, 3]:
            iota = self.get_iota(n_episodes)
        initial_points = []
        pis = []
        k = 0
        while k * self.max_steps < n_episodes:
            k += 1
            if self.random_reset:
                s = self.env.states[np.random.randint(self.env.n_states)]
            else:
                s = s0
            initial_points.append(copy(s))
            self.env.reset(s, reset_rewards=False)
            for h in range(self.max_steps):
                a_index = self._get_action(h, s)
                r, new_s = self.env.step(self.env.actions[self.env.states_indices[s]][a_index])
                next_v = self.V[h + 1][new_s] if h < self.max_steps - 1 else 0
                self.n[h][s][a_index] += 1
                t = self.n[h][s][a_index]
                lr = (self.max_steps + 1) / (self.max_steps + t)
                if self.iota_type == 2:
                    iota = np.log(self.env.n_states * self.env.n_actions * np.sqrt(t) / self.delta)
                if self.bonus == "hoeffding":
                    b = self.c * np.sqrt(self.max_steps**3 * iota / t)
                elif self.bonus == "bernstein":
                    self.mu[h][s][a_index] += next_v
                    self.sigma[h][s][a_index] += next_v ** 2
                    new_beta = min(
                        self.c1 * (np.sqrt(self.max_steps / t * (((self.sigma[h][s][a_index] - self.mu[h][s][a_index])**2 / t) + self.max_steps) * iota) + np.sqrt(self.max_steps ** 7 * self.env.n_actions * self.env.n_states) * iota / t),
                        self.c2 * np.sqrt(self.max_steps ** 3 * iota / t)
                    )
                    b = (new_beta - (1 - lr) * self.beta[h][s][a_index]) / (2 * lr)
                    self.beta[h][s][a_index] = new_beta
                else:
                    raise ValueError(f"Unknown bonus {self.bonus}")
                self.Q[h][s][a_index] += lr * (r + next_v + b - self.Q[h][s][a_index])
                self.V[h][s] = min(self.max_steps, np.max(self.Q[h][s]))
                s = self.env.s
            pis.append(copy([{s: np.argmax(self.Q[h][s]) for s in self.env.states} for h in range(self.max_steps)]))
        return initial_points, pis

    def get_iota(self, n_episodes):
        ret = self.env.n_states * self.env.n_actions / self.delta
        if self.iota_type == 1:
            ret *= n_episodes
        elif self.iota_type == 3:
            ret *= np.log(n_episodes)
        return np.log(ret)

    def get_mu0(self):
        if self.bonus == "hoeffding":
            return None
        else:
            return {h: {s: np.zeros(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}
                    for h in range(self.max_steps)}


class UCBQLRM(UCBQL):
    def __init__(self, env: BaseEnvironment, lr=0.1, eps=0.2, max_steps=100, bonus="hoeffding",
                 c=2, delta=0.05, random_reset=True, iota_type=3, c1=0.5, c2=0.5,
                 reward_shaping=None, resized_rs_reward=True):
        self.rm = env.rm
        super().__init__(env=env, lr=lr, eps=eps, max_steps=max_steps, bonus=bonus, c=c, c1=c1, c2=c2, delta=delta,
                         random_reset=random_reset, iota_type=iota_type)
        self.reward_shaping = reward_shaping
        self.resized_rs_reward = resized_rs_reward

    def get_q0(self):
        return {u:
                {h:
                    {s: self.max_steps * np.ones(len(self.env.actions[self.env.states_indices[s]]))
                     for s in self.env.states}
                 for h in range(self.max_steps)}
                for u in self.rm.states}

    def get_v0(self):
        return {u:
                {h: {s: 0 for s in self.env.states} for h in range(self.max_steps)}
                for u in self.rm.states}

    def get_mu0(self):
        if self.bonus == "hoeffding":
            return None
        else:
            return {u: {h: {s: np.zeros(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}
                    for h in range(self.max_steps)} for u in self.rm.states}

    def _get_action_rm(self, u, h, s):
        assert self.env.actions[self.env.states_indices[s]] != []
        return np.argmax(self.Q[u][h][s])

    def run(self, s0, n_episodes):
        # TODO: change "n_episodes"'s name
        if self.iota_type in [1, 3]:
            iota = self.get_iota(n_episodes)
        initial_points = []
        pis = []
        k = 0

        while k * self.max_steps < n_episodes:
            k += 1
            if self.random_reset:
                u = self.rm.states[np.random.randint(self.rm.n_states)]
                s = self.env.states[np.random.randint(self.env.n_states)]
            else:
                s, u = s0
            initial_points.append(copy((s, u)))
            self.env.reset(s, reset_rewards=False)
            self.rm.u = u

            for h in range(self.max_steps):
                a_index = self._get_action_rm(u, h, s)
                r_tmp, new_s = self.env.step(self.env.actions[self.env.states_indices[s]][a_index])
                new_u = copy(self.rm.u)
                # print(f"H={h}, (s, u): {s, u}, (s', u'): {new_s, new_u}, r = {r_tmp}")
                self.n[h][s][a_index] += 1
                t = self.n[h][s][a_index]
                lr = (self.max_steps + 1) / (self.max_steps + t)
                if self.iota_type == 2:
                    iota = np.log(self.env.n_states * self.env.n_actions * np.sqrt(t) / self.delta)
                if self.bonus == "hoeffding":
                    b = self.c * np.sqrt(self.max_steps**3 * iota / t)
                elif self.bonus == "bernstein":
                    next_v = self.V[self.rm.u][h + 1][new_s] if h < self.max_steps - 1 else 0
                    self.mu[u][h][s][a_index] += next_v
                    self.sigma[u][h][s][a_index] += next_v ** 2
                    new_beta = min(
                        self.c1 * (np.sqrt(self.max_steps / t * (((self.sigma[u][h][s][a_index] - self.mu[u][h][s][a_index])**2 / t) + self.max_steps) * iota) + np.sqrt(self.max_steps ** 7 * self.env.n_actions * self.env.n_states) * iota / t),
                        self.c2 * np.sqrt(self.max_steps ** 3 * iota / t)
                    )
                    b = (new_beta - (1 - lr) * self.beta[u][h][s][a_index]) / (2 * lr)
                    self.beta[u][h][s][a_index] = new_beta
                else:
                    raise ValueError(f"Unknown bonus {self.bonus}")

                for u_ in self.rm.states:
                    self.rm.u = u_
                    r, new_u_ = self.rm.step(s, new_s, perform_transition=False)
                    next_v = self.V[new_u_][h + 1][new_s] if h < self.max_steps - 1 else 0
                    if self.reward_shaping is not None:
                        rs_bonus = self.reward_shaping.get_bonus(u_, new_u_)
                        r += rs_bonus
                        if self.resized_rs_reward:
                            r = (r - self.reward_shaping.min) / (self.reward_shaping.max - self.reward_shaping.min)
                    self.Q[u_][h][s][a_index] += lr * (r + next_v + b - self.Q[u_][h][s][a_index])
                    self.V[u_][h][s] = min(self.max_steps, np.max(self.Q[u_][h][s]))

                s = copy(self.env.s)
                u = new_u
                self.rm.u = new_u

            if self.reward_shaping is not None:
                self.reward_shaping.step()
            pis.append(copy([{(s_, u_): np.argmax(self.Q[u_][h][s_]) for s_ in self.env.states for u_ in self.rm.states}
                             for h in range(self.max_steps)]))
        return initial_points, pis

