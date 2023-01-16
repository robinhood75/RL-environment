import numpy as np


class BaseRewardMachine:
    def __init__(self, u0):
        self.states = None
        self.u = u0
        self.u0 = u0
        self.is_reward_td = False

    def reset(self):
        self.u = self.u0

    def step(self, s, perform_transition=True, state=None):
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

    def step(self, s, perform_transition=True, state=None):
        if s == 0 and self.u == 'RL':
            new_u = 'LR'
            r = 1
        elif s == self.n_states_mdp - 1 and self.u == 'LR' and not self.two_rewards:
            new_u = 'RL'
            r = 1
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

    def step(self, s, perform_transition=True, state=None):
        if s in self.rewards_dict.keys():
            r = self.rewards_dict[s]
        else:
            r = 0
        return r, None


class OneStateRMTD(BaseRewardMachine):
    """transition-dependent"""
    def __init__(self, rewards_dict, u0=None):
        super().__init__(u0=u0)
        self.is_reward_td = True
        self.states = np.unique(np.array(list(rewards_dict.keys())).flatten())
        self.rewards_dict = rewards_dict

    def step(self, s, perform_transition=True, state=None):
        assert state is not None
        if (s, state) in self.rewards_dict.keys():
            r = self.rewards_dict[(s, state)]
        else:
            r = 0
        return r, None
