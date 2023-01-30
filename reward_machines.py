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
    def __init__(self, rm: BaseRewardMachine, phi=None, scaling=None, gamma_=0.9):
        self.rm = rm
        self.phi = self.get_potential(phi)
        self.k = 1
        self.scaling = scaling
        self.gamma = gamma_

    def get_potential(self, phi):
        if phi is not None:
            return phi
        else:
            raise ValueError

    def get_bonus(self, u, new_u):
        ret = self.gamma * self.phi[new_u] - self.phi[u]
        if self.scaling is None:
            pass
        elif self.scaling == 'log':
            ret /= np.log(self.k + 1)
        elif isinstance(self.scaling, float):
            ret *= self.scaling
        else:
            raise ValueError(f"Unknown scaling {self.scaling}")
        return ret

    def step(self):
        self.k += 1
