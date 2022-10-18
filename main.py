from model_based_algorithms import *
from model_free_algorithms import *
from reward_machines import *


if __name__ == '__main__':
    rm = OneStateRM(rewards_dict={(2, 2): 1})
    env = GridWorld(rm, x_max=3, y_max=3, target=(2, 2), walls=[], p=0.9)

    vi = ValueIteration(env, gamma=1., epsilon=0.01, verbose=False)
    vi.start(s0=(0, 0))
    best_actions = vi.run()
    print(f"Best actions (ref): {best_actions}")

    oql = OptimisticQLearning(env, t=100, c=1, max_steps=100)
    oql.run(s0=(0, 0))
    print(f"Q-values (OQL): {oql.Q_est}")
