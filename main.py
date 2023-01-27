import matplotlib.pyplot as plt

from model_based_algorithms import *
from model_free_algorithms import *
from reward_machines import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_names', nargs="*",
                    help="list of env names, where env_name is either vanilla or patrol",
                    required=True)
parser.add_argument('--algo_names', nargs="*",
                    help="list of algo names, where algo is ucbql-h, ucbql-rm-b, etc.",
                    required=True)
parser.add_argument('--n_states', type=int, help="Number of states", default=3)
parser.add_argument('--episode_length', type=int, help="H", default=6)
parser.add_argument('--max_t', type=int, help="Time horizon", default=100000)
parser.add_argument('--n_runs', type=int, help="Number of runs to average on", default=10)
parser.add_argument('--title', type=str, help="Plot title", default="Regret")
args = parser.parse_args()


def _get_algo(algo: str, env_, t, c=1., max_steps=20):
    # TODO: make a proper factory
    if algo == "oql":
        cls = OptimisticQLearning(env=env_, t=t, c=c)
    elif algo == "oqlrm":
        cls = OQLRM(env=env_, t=t, c=c)
    elif algo.startswith("ucbql"):
        try:
            iota_type = int(algo[-1])
        except ValueError:
            iota_type = 3
        if algo.startswith("ucbql-h"):
            cls = UCBQL(env=env_, max_steps=max_steps, c=c, bonus="hoeffding", iota_type=iota_type)
        elif algo.startswith("ucbql-b"):
            cls = UCBQL(env=env_, max_steps=max_steps, c=c, bonus="bernstein", iota_type=iota_type)
        elif algo.startswith("ucbql-rm-h"):
            cls = UCBQLRM(env=env_, max_steps=max_steps, c=c, bonus="hoeffding", iota_type=iota_type)
        elif algo.startswith("ucbql-rm-b"):
            cls = UCBQLRM(env=env_, max_steps=max_steps, c=c, bonus="bernstein", iota_type=iota_type)
        else:
            raise ValueError(f"Unknown algorithm {algo}")
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
        last += v_star[env_copy.states_indices[s]] - _v_fn[s][0]
        ret.append(last)
    return np.array(ret)


def plot_regret(t, env_: BaseEnvironment, s0, n_runs=10, algo="oql", save_to="fig.svg", episode_length=None,
                title=None, opt_gain=None, dp=None, regret_per_episode=True, has_rm_=False):
    regrets = []
    env_copy = deepcopy(env_)
    if has_rm_:
        env_copy = get_cross_product(env_copy, env_copy.rm)

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
    for i, x_scale in enumerate(["log", "linear"]):
        plt.figure(i+1)
        plt.fill_between(times, mean - ci, mean + ci, color='lightblue')
        plt.plot(times, regrets.mean(axis=0), label=algo)
        if title is not None:
            plt.title(title)
        plt.xscale(x_scale)
        if regret_per_episode:
            plt.xlabel("episode nb")
        else:
            plt.xlabel("t")
        plt.ylabel("Regret")


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


def has_rm(algo_, env_name_):
    return env_name_ == "patrol" and algo_.startswith("ucbql-rm")


def get_env_1(n_):
    rewards_dict_ = {0: 0.1/n_, n_-1: 1}
    rm_ = OneStateRM(rewards_dict=rewards_dict_)
    env_ = RiverSwim(rm_, n=n_, p=0.6)

    for k in range(1, n_-1):
        env_.transition_p[k][1] = np.zeros(n_)
        env_.transition_p[k][1][k] = 0.6
        env_.transition_p[k][1][k+1] = 0.35
        env_.transition_p[k][1][k-1] = 0.05

        env_.transition_p[k][-1] = np.zeros(n_)
        env_.transition_p[k][-1][k-1] = 1

    env_.transition_p[-1][-1] = np.zeros(n_)
    env_.transition_p[-1][-1][-2] = 1

    env_.transition_p[-1][1] = np.zeros(n_)
    env_.transition_p[-1][1][-1] = 0.6
    env_.transition_p[-1][1][-2] = 0.4

    return env_


def get_env_patrol(n_, cross_product_=False):
    rm_ = RiverSwimPatrol(u0='LR', n_states_mdp=n_, two_rewards=False)
    env_ = RiverSwim(rm_, n=n_, p=0.6)
    env_.transition_p = get_env_1(n_).transition_p
    if cross_product_:
        env_ = get_cross_product(env_, env_.rm)
    return env_


def get_env(env_name_, n_=3, cross_product_=False):
    if env_name_ == "vanilla":
        env_ = get_env_1(n_)
    elif env_name_ == "patrol":
        env_ = get_env_patrol(n_, cross_product_=cross_product_)
    else:
        raise ValueError(f"Unknown env {env_name_}")
    return env_


if __name__ == '__main__':
    h = args.episode_length
    n = args.n_states
    t = args.max_t
    env_names = args.env_names
    algo_names = args.algo_names
    n_runs = args.n_runs
    plot_title = args.title

    for algo, env_name in zip(algo_names, env_names):
        print(f"Algorithm: {algo}, environment: {env_name}")
        has_rm_ = has_rm(algo, env_name)
        env = get_env(env_name, n, cross_product_=(env_name == "patrol" and not algo.startswith("ucbql-rm")))

        if has_rm_:
            dp = ValueDynamicProgramming(env=get_cross_product(env, env.rm), h=h+1)
        else:
            dp = ValueDynamicProgramming(env=env, h=h + 1)
        v_fn = dp.run()
        dp = [v_fn[k][0] for k in v_fn.keys()]

        if has_rm_:
            print("States of the CP:", get_cross_product(env, env.rm).states)
        else:
            print("States:", env.states)
        print("V*_1:", dp)

        plot_regret(t=int(t),
                    env_=env,
                    s0=None,
                    opt_gain=None,
                    n_runs=n_runs,
                    algo=algo,
                    episode_length=h,
                    title=plot_title,
                    dp=dp,
                    regret_per_episode=True,
                    has_rm_=has_rm_)
    for i in [1, 2]:
        plt.figure(i)
        plt.legend()
    plt.show()

