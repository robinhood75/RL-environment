from model_based_algorithms import *
from model_free_algorithms import *
from reward_machines import *


if __name__ == '__main__':
    rm = RiverSwimPatrol('RL', 5)
    env = RiverSwim(rm, 5, p=0.9)

    q = QRM(env, gamma=0.9, lr=0.1, eps=0.2, max_steps=6000)
    q.run(s0=0, n_episodes=10)
    print(q.Q)
