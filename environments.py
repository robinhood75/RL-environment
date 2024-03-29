import numpy as np
from reward_machines import BaseRewardMachine
from copy import deepcopy
import types


class BaseEnvironment:
    def __init__(self, rm: BaseRewardMachine):
        self.rm = rm
        self.states = None
        self.base_actions = None
        self.actions = None
        self.s = None
        self.states_indices = None
        self.transition_p = None
        self.trajectory = []
        self.rewards = []

    def reset(self, s0, reset_rewards=True):
        self.rm.reset()
        self.s = s0
        self.trajectory = [s0]
        if reset_rewards:
            self.rewards = []

    def start(self, s0):
        self.s = s0
        self.trajectory.append(s0)

    @property
    def n_steps(self):
        return len(self.trajectory)

    @property
    def n_states(self):
        return len(self.states)

    @property
    def n_actions(self):
        return len(self.base_actions)

    def get_next_state(self, action):
        new_state_index = np.random.choice(
            range(self.n_states),
            p=self.transition_p[self.states_indices[self.s]][action]
        )
        return self.states[new_state_index]

    def step(self, action, perform_action=True, new_state=None):
        if new_state is None:
            new_state = self.get_next_state(action)
        reward = self.rm.step(self.s, new_state, perform_transition=perform_action)[0]
        if perform_action:
            self.s = new_state
            self.trajectory.append(new_state)
            self.rewards.append(reward)
        return reward, new_state

    def is_terminal(self, s):
        return self.actions[self.states_indices[s]] == []


class GridWorld(BaseEnvironment):
    def __init__(self, rm, x_max, y_max, walls, p=0.9):
        """
        :param x_max: width of the grid
        :param y_max: length of the grid
        :param walls: list of positions corresponding to walls
        :param target: s of the target
        :param p: the action succeeds with probability p, and the agent doesn't move with probability 1-p
        """
        super().__init__(rm=rm)
        self.x_max = x_max
        self.y_max = y_max
        self.walls = walls
        self.p = p
        self.states = [(x, y) for x in range(x_max) for y in range(y_max) if (x, y) not in walls]
        self.states_indices = {s: i for (i, s) in enumerate(self.states)}
        self.base_actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.actions = [[a for a in self.base_actions
                         if (np.array(a) + np.array(s) < np.array([x_max, y_max])).all()
                         and (np.array(a) + np.array(s) >= np.zeros(2)).all()
                         and tuple((np.array(a) + np.array(s)).tolist()) not in walls]
                        for s in self.states]
        self.transition_p = self.get_transition_p()

    def get_transition_p(self):
        transition_p = [{a: np.zeros(self.n_states) for a in self.actions[s]}
                        for s in range(self.n_states)]
        for s in range(self.n_states):
            actions = self.actions[s]
            next_states = [self.states_indices[tuple(np.array(a) + np.array(self.states[s]))]
                           for a in actions]
            for i, a in enumerate(actions):
                transition_p[s][a][next_states[i]] = self.p
                transition_p[s][a][s] = 1 - self.p
        return transition_p

    def start(self, s0=(0, 0)):
        super().start(s0=s0)


class RiverSwim(BaseEnvironment):
    def __init__(self, rm, n, p=0.8, edge_actions=True):
        super().__init__(rm=rm)
        self.p = p
        self.states = range(n)
        self.states_indices = range(n)
        self.base_actions = [-1, 1]
        if edge_actions:
            self.actions = [[a for a in self.base_actions] for _ in self.states]
        else:
            self.actions = [[a for a in self.base_actions if 0 <= a + s < n]
                            for s in self.states]
        self.transition_p = self.get_transition_p()

    def start(self, s0=0):
        super().start(s0=s0)

    def get_transition_p(self):
        transition_p = [{a: np.zeros(self.n_states) for a in self.actions[s]}
                        for s in self.states]
        for s in self.states:
            actions = self.actions[s]
            next_states = [a + s for a in actions]
            for i, a in enumerate(actions):
                if next_states[i] == -1:
                    transition_p[s][a][0] = 1
                elif next_states[i] == len(self.states):
                    transition_p[s][a][-1] = 1
                else:
                    transition_p[s][a][next_states[i]] = self.p
                    transition_p[s][a][s] = 1 - self.p
        return transition_p


def get_cross_product(env: BaseEnvironment, rm: BaseRewardMachine):
    """Make sure that env has been reset before"""
    assert env.rewards == []
    cp = deepcopy(env)
    cp.states = [(s, u) for s in env.states for u in rm.states]
    cp.states_indices = {s: i for i, s in enumerate(cp.states)}
    cp.actions = [env.actions[env.states_indices[s[0]]] for s in cp.states]

    transition_p = [{a: np.zeros(cp.n_states) for a in cp.actions[i]}
                    for i, s in enumerate(cp.states)]
    for i, s in enumerate(cp.states):
        for a in env.actions[env.states_indices[s[0]]]:
            for j, new_s in enumerate(cp.states):
                rm.u = s[1]
                _, new_u = rm.step(s[0], new_s[0], perform_transition=False)
                if new_s[1] == new_u:
                    transition_p[i][a][j] = env.transition_p[env.states_indices[s[0]]][a][env.states_indices[new_s[0]]]
    cp.transition_p = transition_p

    def new_step_fn(self, action, perform_action=True, new_state=None):
        if new_state is None:
            new_state = self.get_next_state(action)
        self.rm.u = self.s[1]
        reward = self.rm.step(self.s[0], new_state[0], perform_transition=perform_action)[0]
        if perform_action:
            self.s = new_state
            self.trajectory.append(new_state)
            self.rewards.append(reward)
        return reward, new_state

    functype = types.MethodType
    cp.step = functype(new_step_fn, cp)

    return cp


