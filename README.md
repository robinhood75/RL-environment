# RL-environment
This is the repository used to generate the experiments of my master's thesis in the DELTA research group at the University of Copenhagen. It implements the following RL algorithms:
UCB Q-Learning (Jin et al., 2018), our algorithm (UCB Q-Learning for Reward Machines), UCBVI (Azar et al., 2017), UCRL2 for the average-reward setting (Jacksch et al., 2010),
Optimistic Q-Learning (Wei et al., 2020).  
The syntax that should be used to generate an experiment is the following: 

<code>python main.py --env_names \<k environment names separated by space\> --n_states \<number of states of the environment\> --episode_length \<episode length\> --algo_names \<k algo names separated by space\> --max_t \<T\> --n_runs \<number of runs\>
--title \<title of plot\> --reward_shaping (optional) \<k reward shaping modes (str) separated by space\> </code>

where <code>env_names</code> can be chosen from <code>"vanilla"</code> for a regular RiverSwim, <code>"patrol"</code> for a RiverSwim with Patrol Reward Machine,
<code>"patrol\<an int\>"</code> for a RiverSwim with Patrol4, Patrol6, Patrol8... Reward Machine, or <code>"gridworld"</code> for
a GridWorld environment with 3-states Reward Machine.   
<code>algo_names</code> can be chosen from <code>"ucbql-\<suffix\>"</code> for UCB Q-Learning
where <code>\<suffix\></code> is <code>"h-\<int\>"</code> or <code>"b-\<int\>"</code> for Hoeffding/Bernstein and <code>\<int\></code> is a bonus type that can be either 1, 2, 3, <code>"ucbql-rm-\<suffix\>"</code>,
for UCB Q-Learning for RMs, <code>"ucbvi"</code> for UCBVI, <code>"UCRL2"</code> for UCRL2, or <code>"optql"</code> for Optimistic Q-Learning.   
<code>"reward_shaping"</code> can be chosen from <code>None</code> (no reward shaping), <code>\<float\></code> for a potential function $\Phi =$ <code>\<float\></code> $\cdot V^{\*, \text{RM}}$,
<code>"s_dependent"</code> for $\Phi = V^{*, \text{RM}} / N_h(s, a)$, or <code>log</code> for $\Phi = V^{\*, \text{RM}} / \log(k)$.
