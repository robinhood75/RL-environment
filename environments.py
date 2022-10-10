import numpy as np


class BaseEnvironment:
    def __init__(self):
        self.states = None
        self.actions = None
        self.s = None
        self.states_indices = None
        self.transition_p = None
        self.trajectory = []
        self.rewards = []

    def start(self, s0):
        self.s = s0
        self.trajectory.append(s0)

    def step(self, policy=None, action=None, perform_action=None):
        raise NotImplementedError

    @property
    def n_steps(self):
        return len(self.trajectory)

    @property
    def n_states(self):
        return len(self.states)


class GridWorld(BaseEnvironment):
    def __init__(self, x_max, y_max, target, walls, t_max, p=0.8):
        """
        :param x_max: width of the grid
        :param y_max: length of the grid
        :param walls: list of positions corresponding to walls
        :param target: s of the target
        :param t_max: max number of steps allowed in an episode
        :param p: the action succeeds with probability p, and the agent doesn't move with probability 1-p
        """
        super().__init__()
        self.x_max = x_max
        self.y_max = y_max
        self.walls = walls
        self.target = target
        self.t_max = t_max
        self.p = p
        self.states = [np.array([x, y]) for x in range(x_max) for y in range(y_max) if (x, y) not in walls]
        self.states_indices = {tuple(s.tolist()): i for (i, s) in enumerate(self.states)}
        base_actions = [np.array(a) for a in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        self.actions = [[a for a in base_actions
                         if (a + s < np.array([x_max, y_max])).all()
                         and (a + s >= np.zeros(2)).all()
                         and (a + s).tolist() not in walls
                         and not (s == target).all()]
                        for s in self.states]
        self.transition_p = self.get_transition_p()

    def get_transition_p(self):
        transition_p = [{tuple(a.tolist()): np.zeros(self.n_states) for a in self.actions[s]}
                        for s in range(self.n_states)]
        for s in range(self.n_states):
            actions = self.actions[s]
            next_states = [self.states_indices[tuple((a + self.states[s]).tolist())] for a in actions]
            for i, a in enumerate(actions):
                transition_p[s][tuple(a.tolist())][next_states[i]] = self.p
                transition_p[s][tuple(a.tolist())][s] = 1 - self.p
        return transition_p

    def start(self, s0=(0, 0)):
        super().start(s0=s0)

    def step(self, policy=None, action=None, perform_action=True, new_state=None):
        """
        Perform action sampled from policy.
        :param policy: list of actions
        :param action: if not specified, sampled from the policy
        :param perform_action: actually perform action
        :param new_state: is not specified, sampled from p(.|s,a)
        :return: reward
        """
        if action is None:
            assert policy is not None, "Specify either action or policy"
            action = np.random.choice(self.actions[self.states_indices[tuple(self.s.tolist())]], p=policy[tuple(self.s.tolist())])
        if new_state is None:
            new_state = self.get_next_state(action)
        reward = int((new_state == self.target).all())
        if perform_action:
            self.s = new_state
            self.trajectory.append(new_state)
            self.rewards.append(reward)
        return reward

    def get_next_state(self, action):
        new_state_index = np.random.choice(
            range(self.n_states),
            p=self.transition_p[self.states_indices[tuple(self.s.tolist())]][tuple(action.tolist())]
        )
        return self.states[new_state_index]

    def visualize(self):
        """Visualize the grid together with the trajectory"""
        raise NotImplementedError


class RiverSwim(BaseEnvironment):
    def __init__(self, n, p=0.8):
        super().__init__()
        self.p = p
        self.states = range(n)
        base_actions = [-1, 1]
        self.actions = [[a for a in base_actions if 0 <= a + s < n]
                        for s in self.states]
        self.target = self.states[-1]
        self.transition_p = self.get_transition_p()

    def start(self, s0=0):
        super().start(s0=s0)

    def step(self, policy=None, action=None, perform_action=None, new_state=None):
        if action is None:
            assert policy is not None, "Specify either action or policy"
            action = np.random.choice(self.actions[self.s], p=policy[self.s])
        if new_state is None:
            new_state = self.get_next_state(action)
        reward = int((new_state == self.target).all())
        if perform_action:
            self.s = new_state
            self.trajectory.append(new_state)
            self.rewards.append(reward)
        return reward

    def get_transition_p(self):
        transition_p = [{a: np.zeros(self.n_states) for a in self.actions[s]}
                        for s in self.states]
        for s in self.states:
            actions = self.actions[s]
            next_states = [a + s for a in actions]
            for i, a in enumerate(actions):
                transition_p[a][next_states[i]] = self.p
                transition_p[a][next_states[i]] = 1 - self.p
        return transition_p

    def get_next_state(self, action):
        new_state_index = np.random.choice(
            range(self.n_states),
            p=self.transition_p[self.s][action]
        )
        return self.states[new_state_index]
