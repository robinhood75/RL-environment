from environments import BaseEnvironment, GridWorld, RiverSwim
import numpy as np


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

