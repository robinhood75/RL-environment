from online_algorithms import *


if __name__ == '__main__':
    # env = GridWorld(x_max=3,
    #                 y_max=3,
    #                 target=(2, 2),
    #                 walls=[],
    #                 t_max=100,
    #                 p=0.9)
    env = RiverSwim(10, p=0.9)
    print(env.transition_p)
    vi = ValueIteration(env, epsilon=0.001, gamma=0.9)
    policy = vi.run()
    print(policy)
    print(env.states)
