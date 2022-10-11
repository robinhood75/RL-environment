import numpy as np
from reward_machines import BaseRewardMachine


class BaseEnvironment:
    def __init__(self, rm: BaseRewardMachine):
        self.rm = rm
        self.states = None
        self.actions = None
        self.s = None
        self.states_indices = None
        self.transition_p = None
        self.target = None
        self.trajectory = []
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

    def get_next_state(self, action):
        new_state_index = np.random.choice(
            range(self.n_states),
            p=self.transition_p[self.s][action]
        )
        return self.states[new_state_index]

    def step(self, policy=None, action=None, perform_action=None, new_state=None):
        if action is None:
            assert policy is not None, "Specify either action or policy"
            action = np.random.choice(self.actions[self.s], p=policy[self.s])
        if new_state is None:
            new_state = self.get_next_state(action)
        reward = self.rm.step(new_state)
        if perform_action:
            self.s = new_state
            self.trajectory.append(new_state)
            self.rewards.append(reward)
        return reward


class GridWorld(BaseEnvironment):
    def __init__(self, rm, x_max, y_max, target, walls, t_max, p=0.8):
        """
        :param x_max: width of the grid
        :param y_max: length of the grid
        :param walls: list of positions corresponding to walls
        :param target: s of the target
        :param t_max: max number of steps allowed in an episode
        :param p: the action succeeds with probability p, and the agent doesn't move with probability 1-p
        """
        super().__init__(rm=rm)
        self.x_max = x_max
        self.y_max = y_max
        self.walls = walls
        self.target = target
        self.t_max = t_max
        self.p = p
        self.states = [(x, y) for x in range(x_max) for y in range(y_max) if (x, y) not in walls]
        self.states_indices = {s: i for (i, s) in enumerate(self.states)}
        base_actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.actions = [[a for a in base_actions
                         if (np.array(a) + np.array(s) < np.array([x_max, y_max])).all()
                         and (np.array(a) + np.array(s) >= np.zeros(2)).all()
                         and tuple((np.array(a) + np.array(s)).tolist()) not in walls
                         and not s == target]
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
    def __init__(self, rm, n, p=0.8):
        super().__init__(rm=rm)
        self.p = p
        self.states = range(n)
        self.states_indices = range(n)
        base_actions = [-1, 1]
        self.actions = [[a for a in base_actions if 0 <= a + s < n]
                        for s in self.states]
        self.target = self.states[-1]
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
                transition_p[s][a][next_states[i]] = self.p
                transition_p[s][a][s] = 1 - self.p
        return transition_p
