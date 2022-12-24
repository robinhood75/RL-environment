from environments import *
import numpy as np
from copy import copy


class ValueIteration:
    def __init__(self, env: BaseEnvironment, epsilon, gamma=1, verbose=True):
        self.verbose = verbose
        self.env = env
        self.gamma = self.validate_gamma(gamma)
        self.last_v_fn = self.get_v0(env.states)
        self.v_fn = self.get_v1(env.states, gamma)
        self.stopping_crit = epsilon * (1 - gamma) / (2 * gamma) if gamma < 1 else epsilon
        self.n_steps = 0

    def start(self, s0):
        self.env.start(s0)

    def step(self):
        self.last_v_fn = self.v_fn.copy()
        self.v_fn = {s: self.step_s(s) for s in self.env.states}
        self.n_steps += 1
        if self.verbose:
            print(f"Value fn after {self.n_steps} steps: \n"
                  f"{self.v_fn}")

    def step_s(self, s, return_argmax=False):
        """
        :param s: state
        :param return_argmax: if True, returns argmax over a instead of max over a
        """
        self.env.s = s
        s_index = self.env.states_indices[s]
        actions = self.env.actions[s_index]
        v = np.array(list(self.v_fn.values()))
        r = np.array([[self.env.step(action=a, new_state=x, perform_action=False)[0] for x in self.env.states]
                      for a in actions])
        tot = self.gamma * v + r if actions else np.array([])
        values = [tot[i] @ self.env.transition_p[self.env.states_indices[s]][a]
                  for i, a in enumerate(actions)]
        if return_argmax:
            if values:
                a = np.argmax(values)
                return actions[a]
            else:
                return None
        else:
            return np.max(values) if values else 0

    @staticmethod
    def validate_gamma(gamma):
        if gamma > 1:
            raise ValueError(f"{gamma} should be <= 1")
        return gamma

    @staticmethod
    def get_v0(states):
        values = np.random.random(len(states))
        return {s: v for v, s in zip(values, states)}

    @staticmethod
    def get_v1(states, gamma):
        n_states = len(states)
        if gamma == 1:
            values = np.random.random(n_states)
        else:
            values = 1 / (1 - gamma) * np.ones(n_states)
        return {s: v for v, s in zip(values, states)}

    def run(self):
        to_val = lambda v: np.array(list(v.values()))
        if self.gamma < 1:
            compare = lambda v0, v1: np.sqrt(np.sum((to_val(v0) - to_val(v1))**2)) < self.stopping_crit
        else:
            compare = lambda v0, v1: np.max(to_val(v1) - to_val(v0)) - np.min(to_val(v1) - to_val(v0)) < self.stopping_crit

        while not compare(self.v_fn, self.last_v_fn):
            self.step()

        if self.verbose:
            print(f"VI stopped after {self.n_steps} iterations")
        return [self.step_s(s, return_argmax=True) for s in self.env.states]


class ValueDynamicProgramming:
    """For finite-horizon MDPs"""
    def __init__(self, env: BaseEnvironment, h):
        self.env = env
        self.h = h
        self.v_fn = self.get_v0(self.env, h)

    @staticmethod
    def get_v0(env: BaseEnvironment, h):
        v = {s: np.zeros(h) for s in env.states}
        # for s in v.keys():
        #     v[s][-1] = env.rm.step(s, perform_transition=False)[0]
        return v

    def run(self):
        for h in np.flip(np.arange(self.h - 1)):
            for s in self.env.states:
                self.env.s = s
                actions = self.env.actions[self.env.states_indices[s]]
                rewards = np.array([[self.env.step(a, new_state=x, perform_action=False)[0] for x in self.env.states]
                                    for a in actions])
                exp_r = [self.env.transition_p[self.env.states_indices[s]][a] @
                         np.array([self.v_fn[x][h + 1] + rewards[a_index][x_index]
                                   for x_index, x in enumerate(self.env.states)])
                         for a_index, a in enumerate(actions)]
                self.v_fn[s][h] = np.max(exp_r)
        return self.v_fn


class UCRL2:
    def __init__(self, env: BaseEnvironment, delta):
        self.env = env
        self.delta = delta
        self.n = self.get_n0()
        self.v = self.get_v0()
        self.r = self.get_r0()
        self.counts = self.get_init_counts()
        self.t0 = 0
        self.t = 1
        self.k = 1

    def run(self, n_episodes, s0):
        s = s0
        while self.t < n_episodes:
            tk = self.t
            for s in self.env.states:
                self.n[s] += self.v[s]
            r_est = {s: self.r[s] / np.array([max(1, val) for val in self.n[s]]) for s in self.env.states}
            p_est = {s: {a: self.counts[s][a] / max(1, self.n[s][i])
                         for i, a in enumerate(self.env.actions[self.env.states_indices[s]])}
                     for s in self.env.states}
            beta, beta_p = self.get_bonuses()
            pi = self.evi(p_est, r_est, beta, beta_p, 1 / np.sqrt(tk))
            self.v = self.get_v0()
            while not self._stopping_criterion_ucrl2(s, pi):
                self.env.s = s
                action_idx = pi[s]
                a = self.env.actions[s][action_idx]
                r, new_s = self.env.step(a, perform_action=True)
                self.r[s][action_idx] += r
                self.v[s][action_idx] += 1
                self.counts[s][a][self.env.states_indices[new_s]] += 1
                self.t += 1
                s = new_s

    def _stopping_criterion_ucrl2(self, s, pi):
        return self.v[s][pi[s]] >= max(1, self.n[s][pi[s]])

    def get_bonuses(self):
        beta = {s: np.sqrt(14 * self.env.n_states / np.array([max(1, val) for val in self.n[s]]) * np.log(2 * self.env.n_actions * self.t / self.delta))
                for s in self.env.states}
        beta_p = {s: np.sqrt(3.5 / np.array([max(1, val) for val in self.n[s]]) * np.log(2 * self.env.n_states * self.env.n_actions * self.t / self.delta))
                  for s in self.env.states}
        return beta, beta_p

    def get_init_counts(self):
        return {s: {a: np.zeros(self.env.n_states) for a in self.env.actions[self.env.states_indices[s]]}
                for s in self.env.states}

    def get_r0(self):
        return {s: np.zeros(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}

    def get_v0(self):
        return {s: np.zeros(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}

    def get_n0(self):
        return {s: np.zeros(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}

    def evi(self, p_est, r_est, beta, beta_p, eps, n_max=1000):
        #TODO: replace tk with eps=0.0
        v = self.get_v0_evi()
        last_v = v.copy()
        n = -1
        mu = {s: r_est[s] + beta_p[s] for s in self.env.states}
        p_p = {s: {a: None for a in self.env.actions[self.env.states_indices[s]]}
               for s in self.env.states}

        while not UCRL2._evi_stopping_criterion(v, last_v, eps, n) and n < n_max:
            n += 1
            new_v = {}
            for s in self.env.states:
                for i, a in enumerate(self.env.actions[self.env.states_indices[s]]):
                    p_p[s][a] = self._get_p_p(v, beta[s][i], p_est[s][a])
            for s in self.env.states:
                new_v[s] = np.max([mu[s] +
                                   np.array([np.sum([p_p[s][a][i] * v[x] for i, x in enumerate(self.env.states)])
                                             for a in self.env.actions[self.env.states_indices[s]]])
                                   ])
            v = new_v.copy()
            last_v = v.copy()

        pi = {s: np.argmax([mu[s] +
                            np.array([np.sum([p_p[s][a][i] * v[x] for i, x in enumerate(self.env.states)])
                                      for a in self.env.actions[self.env.states_indices[s]]])
                            ])
              for s in self.env.states}
        return pi

    def _get_p_p(self, v, beta, p_est):
        """
        Algorithm 3 in Sadegh's lecture notes
        beta, p_est are specific to (s, a)
        """
        tmp = np.flip(np.argsort(list(v.values())))
        sorted_s = [list(v.keys())[idx] for idx in tmp]
        q = np.zeros(self.env.n_states)
        q[0] = min(1, p_est[sorted_s[0]] + beta / 2)
        q[1:] = p_est[sorted_s[1:]]
        l_ = self.env.n_states - 1
        while q.sum() > 1:
            q[sorted_s[l_]] = max(0, 1 - q.sum() + q[sorted_s[l_]])
            l_ -= 1
        return q

    @staticmethod
    def _evi_stopping_criterion(v, last_v, eps, n):
        if n < 0:
            return False
        else:
            diff = np.array([val - last_val for val, last_val in zip(list(v.values()), list(last_v.values()))])
            return np.max(diff) - np.min(diff) < eps

    def get_v0_evi(self):
        return {s: 0 for s in self.env.states}


class UCBVI:
    """This algorithm assumes that the reward function is known. So before running it we compute estimates."""
    def __init__(self, env: BaseEnvironment, episode_length, bonus="hoeffding", delta=0.05, c=1):
        self.env = env
        self.episode_length = episode_length
        self.bonus = bonus
        self.delta = delta
        self.c = c
        self.n = self.get_n0()
        self.n_p = self.get_n_p0()
        self.q = self.get_q0()
        self.counts = self.get_init_counts()
        self.exp_r = self.estimate_rewards(self.env, n_epochs=10000)

    def run(self, s0, n_episodes, random_reset=True):
        initial_points = []
        k = 0
        while k * self.episode_length < n_episodes:
            k += 1
            if random_reset:
                s = self.env.states[np.random.randint(self.env.n_states)]
            else:
                s = s0
            initial_points.append(copy(s))
            self.update_ucb_q_values(n_episodes)
            for h in range(self.episode_length):
                a_idx = np.argmax(self.q[h][s])
                a = self.env.actions[self.env.states_indices[s]][a_idx]
                self.env.s = s
                r, new_s = self.env.step(a)
                self.n[s][a_idx] += 1
                self.counts[s][a][self.env.states_indices[new_s]] += 1
                self.n_p[h][s][a_idx] += 1
                s = new_s
        return initial_points

    def update_ucb_q_values(self, k):
        set_ = {s: [a for i, a in enumerate(self.env.actions[self.env.states_indices[s]]) if self.n[s][i] > 0]
                for s in self.env.states}
        p_hat = {s: {a: self.counts[s][a] / self.n[s][i] for i, a in enumerate(v)} for s, v in set_.items()}
        v = {s: 0 for s in self.env.states}
        for h in np.flip(range(self.episode_length)):
            for s in self.env.states:
                for i, a in enumerate(self.env.actions[self.env.states_indices[s]]):
                    if a in set_[s]:
                        b = self.get_bonus_term(t=self.n[s][i], n=self.n[s][i])
                        self.q[h][s][i] = np.min([self.q[h][s][i], self.episode_length,
                                                  self.exp_r[s][i] + p_hat[s][a] @ np.array(list(v.values())) + self.c * b])
                    else:
                        self.q[h][s][i] = self.episode_length
                v[s] = np.max(self.q[h][s])

    def get_init_counts(self):
        return {s: {a: np.zeros(self.env.n_states) for a in self.env.actions[self.env.states_indices[s]]}
                for s in self.env.states}

    def get_r0(self):
        return {s: np.zeros(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}

    def get_q0(self):
        return {h:
                {s: np.zeros(len(self.env.actions[self.env.states_indices[s]]))
                 for s in self.env.states}
                for h in range(self.episode_length)}

    def get_n0(self):
        return {s: np.zeros(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}

    def get_n_p0(self):
        return {h: {s: np.zeros(len(self.env.actions[self.env.states_indices[s]])) for s in self.env.states}
                for h in range(self.episode_length)}

    def get_bonus_term(self, t, n, p_hat=None, v=None, n_p=None):
        l_ = np.log(5 * self.env.n_states * self.env.n_actions * t / self.delta)
        if self.bonus == "hoeffding":
            b = 7 * self.episode_length * l_ / np.sqrt(n)
        elif self.bonus == "bernstein":
            assert p_hat is not None and v is not None and n_p is not None
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown bonus {self.bonus}")
        return b

    @staticmethod
    def estimate_rewards(env: BaseEnvironment, n_epochs=100000):
        r_tot = {s: np.zeros(len(env.actions[env.states_indices[s]])) for s in env.states}
        for n in range(n_epochs):
            for s in env.states:
                for i, a in enumerate(env.actions[env.states_indices[s]]):
                    env.s = s
                    r_tot[s][i] += env.step(action=a, perform_action=False)[0] / n_epochs
        print(f"Estimates computed. r_est: {r_tot}")
        return r_tot
