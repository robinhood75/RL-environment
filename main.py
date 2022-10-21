import matplotlib.pyplot as plt
from model_based_algorithms import *
from model_free_algorithms import *
from reward_machines import *


def plot_regret_oql(t_array, env_: BaseEnvironment, s0, opt_gain, n_runs=3, algo="oql"):
    """
    :param t_array: array of values for T
    :param env_:
    :param s0:
    :param opt_gain: optimal gain for this environment
    :param n_runs: number of runs to average regrets on
    :param algo: either "oql" for Optimistic Q-Learning, or "oqlrm" for Optimistic Q-Learning with reward machines
    """
    regrets = []
    for run_nb in range(n_runs):
        regrets.append([])
        for i, t in enumerate(t_array):
            t = int(t)
            env_.reset(s0, reset_rewards=True)
            if algo == "oql":
                oql = OptimisticQLearning(env_, t=t, c=1)
            elif algo == "oqlrm":
                oql = OQLRM(env_, t=t, c=1)
            else:
                raise ValueError(f"Unknown algorithm {algo}")
            oql.run(s0=s0)

            regrets[-1].append(t * opt_gain - np.sum(env_.rewards))

    regrets = np.mean(regrets, axis=0)
    plt.loglog(t_array, regrets)
    plt.show()


if __name__ == '__main__':
    n = 6
    rm = RiverSwimPatrol(u0="RL", n_states_mdp=n)
    env = RiverSwim(rm, n=n, p=0.9)

    t_array = np.logspace(4, 6.5, 20)
    plot_regret_oql(t_array, env_=env, s0=0, opt_gain=0.2, algo="oqlrm", n_runs=2)
