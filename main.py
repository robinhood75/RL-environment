import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from model_based_algorithms import *
from model_free_algorithms import *
from reward_machines import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, help="Plotting regret is the only implemented task for now",
                    default="plot_regret")
parser.add_argument('--env', type=str, help="Chosen environment: either gridworld or riverswim", default="riverswim")
parser.add_argument('--n_states', type=int, help="Number of states. For gridworld, specify the square of an integer",
                    default=6)
parser.add_argument('--rm', type=str, help="Chosen reward machine: either one_state or patrol", default="one_state")
parser.add_argument('--target', help="Target if the reward is 0 everywhere except at a given target state",
                    default=None)
parser.add_argument('--algo', type=str, help="Either oql or oqlrm", default="oqlrm")
parser.add_argument('--n_runs', type=int, help="Number of runs to average on", default=3)
parser.add_argument('--verbose', type=bool, default=True)
args = parser.parse_args()

if args.task == 'plot_regret':
    pass
else:
    raise ValueError(f"Unknown task {args.task}")


def _get_algo(algo: str, env_, t, c=1., max_steps=20):
    # TODO: make a proper factory (if possible)
    if algo == "oql":
        cls = OptimisticQLearning(env=env_, t=t, c=c)
    elif algo == "oqlrm":
        cls = OQLRM(env=env_, t=t, c=c)
    elif algo == "ucbql-h":
        cls = UCBQL(env=env_, max_steps=max_steps, c=c, bonus="hoeffding")
    elif algo == "ucbql-b":
        cls = UCBQL(env=env_, max_steps=max_steps, c=c, bonus="bernstein")
    elif algo == "ucrl2":
        cls = UCRL2(env=env_, delta=0.05)
    elif algo == "ucbvi":
        cls = UCBVI(env=env_, episode_length=max_steps)
    else:
        raise ValueError(f"Unknown algorithm {algo}")
    return cls


def _get_regret(t, _opt_gain, _rewards):
    ret = []
    last = 0
    for k in range(t):
        last += _opt_gain - _rewards[k]
        ret.append(last)
    return ret


def _get_regret_finite_h(v_star, pis, initial_points, env_copy: BaseEnvironment, _h):
    ret = []
    last = 0
    for i, s in enumerate(initial_points):
        _dp = ValueDynamicProgramming(env_copy, _h+1)
        _v_fn = _dp.run(pis[i])
        last += v_star[s] - _v_fn[s][0]
        ret.append(last)
    return np.array(ret)


def plot_regret(t, env_: BaseEnvironment, s0, n_runs=10, algo="oql", save_to="fig.svg", episode_length=None,
                title=None, opt_gain=None, dp=None, regret_per_episode=True):
    regrets = []
    env_copy = deepcopy(env_)
    for run_nb in range(n_runs):
        print(f"Run {run_nb+1}/{n_runs}")
        env_.reset(s0=s0, reset_rewards=True)
        oql = _get_algo(algo=algo, env_=env_, t=t, c=0.1, max_steps=episode_length)
        if algo.startswith("ucbql"):
            initial_points, pis = oql.run(s0=s0, n_episodes=t)
        else:
            initial_points = oql.run(s0=s0, n_episodes=t)

        if regret_per_episode:
            assert dp is not None
            regret = _get_regret_finite_h(dp, pis, initial_points, env_copy, episode_length)
        else:
            if opt_gain is None:
                assert dp is not None and episode_length is not None
                opt_gains = [g/episode_length for g in dp]
                s0_counts = np.unique(initial_points, return_counts=True)[1]
                s0_frequencies = s0_counts / s0_counts.sum(); print(s0_frequencies)
                opt_gain_tmp = s0_frequencies @ opt_gains
            else:
                opt_gain_tmp = opt_gain
            regret = np.array(_get_regret(t, opt_gain_tmp, env_.rewards))
        regrets.append(regret)

    regrets = np.array(regrets)
    std, mean = np.std(regrets, axis=0), np.mean(regrets, axis=0)
    ci = 1.96 * std / np.sqrt(regrets.shape[0])
    times = range(regrets.shape[1])
    for x_scale in ["log", "linear"]:
        plt.fill_between(times, mean - ci, mean + ci, color='lightblue')
        plt.plot(times, regrets.mean(axis=0), color='blue')
        if title is not None:
            plt.title(title)
        plt.xscale(x_scale)
        if regret_per_episode:
            plt.xlabel("episode nb")
        else:
            plt.xlabel("t")
        plt.ylabel("Regret")
        plt.show()

        if save_to:
            plt.savefig(x_scale + "_" + save_to)


def plot_regret_oql_multiple_t(t_array, env_: BaseEnvironment, s0, opt_gain, n_runs=3, algo="oql", save_to="fig.svg"):
    """
    Plot regret of optimistic Q-learning algorithms in function of length of episode.

    :param t_array: array of values for T
    :param env_:
    :param s0:
    :param opt_gain: optimal gain for this environment
    :param n_runs: number of runs to average regrets on
    :param algo: either "oql" for Optimistic Q-Learning, or "oqlrm" for Optimistic Q-Learning with reward machines
    :param save_to:
    """
    regrets = []
    for run_nb in range(n_runs):
        regrets.append([])
        for i, t in enumerate(t_array):
            t = int(t)
            env_.reset(s0, reset_rewards=True)
            oql = _get_algo(algo=algo, env_=env_, t=t)
            oql.run(s0=s0, n_episodes=t)
            regrets[-1].append(t * opt_gain - np.sum(env_.rewards))

    regrets = np.mean(regrets, axis=0)
    plt.plot(t_array, regrets)
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel("T")
    plt.ylabel("R")
    plt.savefig(save_to)
    plt.show()


def get_env_1():
    n_ = 3
    rewards_dict_ = {0: 0.05, n-1: 1}
    rm_ = OneStateRM(rewards_dict=rewards_dict_)
    env_ = RiverSwim(rm_, n=n_, p=0.4)

    # Get Sadegh's transition probabilities
    env_.transition_p[1][1] = [0.05, 0.6, 0.35]
    env_.transition_p[2][1] = [0, 0.6, 0.4]
    env_.transition_p[2][-1] = [1, 0, 0]
    env_.transition_p[1][-1] = [1., 0, 0]

    return env_


if __name__ == '__main__':
    n = 3
    h = 3
    # rewards_dict = {n-1: 1}
    # rm = OneStateRM(rewards_dict=rewards_dict)
    # env = RiverSwim(rm, n=n, p=0.9)

    env = get_env_1()
    print(env.transition_p)

    dp = ValueDynamicProgramming(env=env, h=h+1)
    v_fn = dp.run(); print(v_fn)
    opt_gain = v_fn[2][0] / h

    dp = [v_fn[k][0] for k in v_fn.keys()]

    plot_regret(t=int(1e6),
                env_=env,
                s0=None,
                opt_gain=None,
                n_runs=20,
                algo="ucbql-h",
                episode_length=h,
                title="UCB-QL (Hoeffding), bonus 3",
                dp=dp,
                regret_per_episode=True)

