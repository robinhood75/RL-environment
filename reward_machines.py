import copy

import numpy as np


class BaseRewardMachine:
    def __init__(self, u0):
        self.states = None
        self.u = u0
        self.u0 = u0

    def reset(self):
        self.u = self.u0

    def step(self, s, next_s, perform_transition=True):
        """
        Transition to new state of RM
        :return: reward, new state
        """
        raise NotImplementedError

    @property
    def n_states(self):
        return len(self.states)


class RiverSwimPatrol(BaseRewardMachine):
    def __init__(self, u0, n_states_mdp, two_rewards=True):
        super().__init__(u0)
        self.n_states_mdp = n_states_mdp
        self.states = ['LR', 'RL']
        self.two_rewards = two_rewards

    def step(self, s, next_s, perform_transition=True, state=None):
        # TODO: simplify disjunction
        if s == 1 and next_s == 0 and self.u == 'RL':
            new_u = 'LR'
            r = 1
        elif s == self.n_states_mdp - 2 and next_s == self.n_states_mdp - 1 and self.u == 'LR':
            if self.two_rewards:
                new_u = 'RL'
                r = 1
            else:
                new_u = 'RL'
                r = 0
        elif s == 0 and self.u == 'RL':
            new_u = 'LR'
            r = 0
        elif s == self.n_states_mdp - 1 and self.u == 'LR':
            new_u = 'RL'
            r = 0
        else:
            new_u = self.u
            r = 0
        if perform_transition:
            self.u = new_u
        return r, new_u


class FourStatesPatrol(BaseRewardMachine):
    def __init__(self, u0, n_states_mdp):
        super().__init__(u0)
        self.n_states_mdp = n_states_mdp
        self.states = ['LR1', 'RL1', 'LR2', 'RL2']

    def step(self, s, next_s, perform_transition=True):
        r = 0
        if s == 1 and next_s == 0:
            if self.u == 'RL1':
                new_u = 'LR2'
            elif self.u == 'RL2':
                new_u = 'LR1'
                r = 1
            else:
                new_u = self.u
        elif s == self.n_states_mdp - 2 and next_s == self.n_states_mdp - 1:
            if self.u == 'LR1':
                new_u = 'RL1'
            elif self.u == 'LR2':
                new_u = 'RL2'
            else:
                new_u = self.u
        else:
            new_u = self.u
        if perform_transition:
            self.u = new_u
        return r, new_u


class LargePatrol(BaseRewardMachine):
    def __init__(self, u0, n_states_mdp, n_states_rm=8):
        super().__init__(u0)
        self.n_states_mdp = n_states_mdp
        self.n_states_rm = n_states_rm
        self.states = self.get_states(n_states_rm)

    @staticmethod
    def get_states(n):
        n_cycles = int(n / 2)
        return [s + str(k) for k in range(1, n_cycles + 1) for s in ['LR', 'RL']]

    def step(self, s, next_s, perform_transition=True):
        r = 0
        if next_s == 0 and s in [0, 1]:
            if self.u.startswith('RL'):
                if 2 * int(self.u[-1]) == self.n_states_rm:
                    new_u = 'LR1'
                    r = s == 1
                else:
                    new_u = 'LR' + str(int(self.u[-1]) + 1)
            else:
                new_u = self.u
        elif self.n_states_mdp - s in [1, 2] and next_s == self.n_states_mdp - 1:
            if self.u.startswith('LR'):
                new_u = 'RL' + self.u[-1]
            else:
                new_u = self.u
        else:
            new_u = self.u

        if perform_transition:
            self.u = new_u

        return r, new_u


class OneStateRM(BaseRewardMachine):
    def __init__(self, rewards_dict, u0=None):
        """:param rewards_dict: dict with keys targets and values the associated rewards"""
        super().__init__(u0=u0)
        self.states = list(rewards_dict.keys())
        self.rewards_dict = rewards_dict

    def step(self, s, next_s, perform_transition=True, state=None):
        if next_s in self.rewards_dict.keys():
            r = self.rewards_dict[next_s]
        else:
            r = 0
        return r, None


class RewardShaping:
    def __init__(self, rm: BaseRewardMachine, v_opt=None, scaling=None, gamma_=0.9, env=None):
        self.rm = rm
        self.scaling = scaling
        self.phi0 = self.get_potential(v_opt)
        self.phi = copy.copy(self.phi0)
        self.k = 1
        self.gamma = gamma_
        self.env = env
        self.min, self.max = self.resized_phi(self.env, self.phi, self.gamma)

    def get_potential(self, v_opt):
        if v_opt is not None:
            if isinstance(self.scaling, float):
                return {k: self.scaling * v for k, v in v_opt.items()}
            else:
                return v_opt
        else:
            raise ValueError

    def get_bonus(self, u, new_u):
        ret = self.gamma * self.phi[new_u] - self.phi[u]
        return ret

    def step(self):
        self.k += 1
        if self.scaling == 'log':
            self.phi = {k: v / np.log(self.k) for k, v in self.phi0.items()}
        elif self.scaling == 'sqrt':
            self.phi = {k: v / np.sqrt(self.k) for k, v in self.phi0.items()}
        if self.scaling in ['log', 'sqrt']:
            self.min, self.max = self.resized_phi(self.env, self.phi, self.gamma)

    @staticmethod
    def resized_phi(env, phi, gamma_):
        min_, max_ = 1e9, -1e9
        for s_index, s in enumerate(env.states):
            env.s = s
            for u in env.rm.states:
                env.rm.u = u
                for a in env.actions[s_index]:
                    r, new_s = env.step(a, perform_action=False)
                    _, new_u = env.rm.step(s, new_s, perform_transition=False)
                    new_r = r + gamma_ * phi[new_u] - phi[u]
                    if new_r < min_:
                        min_ = new_r
                    if new_r > max_:
                        max_ = new_r
        return min_, max_

