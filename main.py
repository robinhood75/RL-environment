from online_algorithms import *
from reward_machines import *


if __name__ == '__main__':
    # rm = OneStateRM({(2, 2): 1})
    # env = GridWorld(x_max=3,
    #                 y_max=3,
    #                 target=(2, 2),
    #                 walls=[],
    #                 t_max=100,
    #                 p=0.9,
    #                 rm=rm)
    rm = RiverSwimPatrol(u0='LR', n_states_mdp=10)
    env = RiverSwim(rm, 10, p=0.9)
    vi = ValueIteration(env, epsilon=0.001, gamma=1)
    policy = vi.run()
    print(policy)
    print(env.states)
