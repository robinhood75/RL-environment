from model_based_algorithms import *
from model_free_algorithms import *
from reward_machines import *


if __name__ == '__main__':
    target = (2, 2)
    rm = OneStateRM({target: 1})
    env = GridWorld(rm, x_max=3, y_max=3, target=target, p=0.9, walls=[])

    q = QLearning(env, gamma=0.9, eps=0.2, max_steps=1000)
    q.run(s0=(0, 0), n_episodes=100)
    print("Q values with QL:", q.Q)

    env.reset(s0=(0, 0))
    vi = ValueIteration(env, epsilon=0.001, gamma=0.9)
    vi.run()
