import matplotlib.pyplot as plt
from model_based_algorithms import *
from model_free_algorithms import *
from reward_machines import *


def plot_regret_oql(t_array, env_: BaseEnvironment, s0, opt_gain, n_runs=3):
    regrets = []
    for run_nb in range(n_runs):
        regrets.append([])
        for i, t in enumerate(t_array):
            t = int(t)
            env_.reset(s0, reset_rewards=True)
            oql = OptimisticQLearning(env_, t=t, c=1)
            oql.run(s0=s0)

            regrets[-1].append(t * opt_gain - np.sum(env_.rewards))

    regrets = np.mean(regrets, axis=0)
    plt.loglog(t_array, regrets)
    plt.show()


if __name__ == '__main__':
    n, eps = 6, 0.
    rm = OneStateRM(rewards_dict={s: 1 if s == n - 1 else -eps for s in range(n)})
    env = RiverSwim(rm, n=n, p=0.9)

    vi = ValueIteration(env, gamma=1., epsilon=0.1, verbose=False)
    vi.start(s0=0)
    best_actions = vi.run()
    print(f"Best actions (ref): {best_actions}")

    t_array = np.logspace(4, 6.5, 20)
    plot_regret_oql(t_array, env_=env, s0=0, opt_gain=0.5)
