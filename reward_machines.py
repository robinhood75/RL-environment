class BaseRewardMachine:
    def __init__(self, u0):
        self.states = None
        self.u = u0

    def step(self, s):
        """
        Transition to new state of RM
        :return: reward
        """
        raise NotImplementedError


class RiverSwimPatrol(BaseRewardMachine):
    def __init__(self, u0, n_states_mdp):
        super().__init__(u0)
        self.n_states_mdp = n_states_mdp
        self.states = ['LR', 'RL']

    def step(self, s):
        if s == 0 and self.u == 'RL':
            self.u = 'LR'
            r = 1
        elif s == self.n_states_mdp - 1 and self.u == 'LR':
            self.u = 'RL'
            r = 1
        else:
            r = 0
        return r


class OneStateRM(BaseRewardMachine):
    def __init__(self, rewards_dict, u0=None):
        """:param rewards_dict: dict with keys targets and values the associated rewards"""
        super().__init__(u0=u0)
        self.states = list(rewards_dict.keys())
        self.rewards_dict = rewards_dict

    def step(self, s):
        if s in self.rewards_dict.keys():
            r = self.rewards_dict[s]
        else:
            r = 0
        return r

