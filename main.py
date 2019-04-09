'''
Description:
    The goal of this assignment is to implement three basic algorithms to solve multi-armed bandit problem.
        1. Epislon-Greedy Alogorithm 
        2. Upper-Confidence-Bound Action Selection
        3. Gradient Bandit Algorithms
    Follow the instructions in code to complete your assignment :)
'''
# import standard libraries
import random
import argparse
import numpy as np

# import others
from env import Gaussian_MAB, Bernoulli_MAB
from algo import EpislonGreedy, UCB, Gradient
from utils import plot

# function map
FUNCTION_MAP = {'e-Greedy': EpislonGreedy, 
                'UCB': UCB,
                'grad': Gradient}
# train function 
def train(args, env, algo):
    reward = np.zeros(args.max_timestep)
    if algo == UCB:
        parameter = args.c
    elif algo == EpislonGreedy:
        parameter = args.epislon
    else:
        parameter = args.alpha

    # start multiple experiments
    for _ in range(args.num_exp):
        # start with new environment and policy
        mab = env(args.num_of_bandits)
        agent = algo(args.num_of_bandits, parameter)
        for t in range(args.max_timestep):
            # choose action first
            a = agent.act(t)

            # get reward from env
            r = mab.step(a)

            # update
            agent.update(a, r)

            # append to result
            reward[t] += r
    
    avg_reward = reward / args.num_exp
    return avg_reward

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-nb", "--num_of_bandits", type=int, 
                        default=10, help="number of bandits")
    parser.add_argument("-algo", "--algo",
                        default="e-Greedy", choices=FUNCTION_MAP.keys(),
                        help="Algorithm to use")
    parser.add_argument("-epislon", "--epislon", type=float,
                        default=0.1, help="epislon for epislon-greedy algorithm")
    parser.add_argument("-c", "--c", type=float,
                        default=1, help="c for UCB")
    parser.add_argument("-alpha", "--alpha", type=float,
                        default=0.4, help="alpha for Gradient")
    parser.add_argument("-max_timestep", "--max_timestep", type=int,
                        default=1000, help="Episode")
    parser.add_argument("-num_exp", "--num_exp", type=int,
                        default=2000, help="Total experiments to run")
    parser.add_argument("-plot", "--plot", action='store_true',
                        help='plot the results')
    parser.add_argument("-runAll", "--runAll", action='store_true',
                        help='run all three algos')
    parser.add_argument("-runEpislon", "--runEpislon", action="store_true",
                        help='run different epislons for e-Greedy')
    parser.add_argument("-runC", "--runC", action="store_true",
                        help='run different c for UCB')
    parser.add_argument("-runBandits", "--runBandits", action="store_true",
                        help='run different numbers of bandits')
    args = parser.parse_args()

    PARAMETER_MAP = {'e-Greedy': ('epislon', args.epislon), 
                     'UCB'     : ('c', args.c),
                     'grad'    : ('alpha', args.alpha)}
    # start training
    # Keep in mind, don't use ~ as the inverter, it bears different meanings.
    runSingle = not (args.runAll or args.runEpislon or args.runC or args.runBandits)
    if runSingle:
        avg_reward = train(args, Gaussian_MAB, FUNCTION_MAP[args.algo])
        if args.plot:
            plot(np.expand_dims(avg_reward, axis=0), [args.algo], [PARAMETER_MAP[args.algo]])
    
    ##############################################################################
    # After you implement all the method, uncomment this part, and then you can  #  
    # use the flag: --runAll to show all the results in a single figure.         #
    ##############################################################################

    if args.runAll:
        _all = ['e-Greedy', 'UCB', 'grad']
        avg_reward = np.zeros([len(_all), args.max_timestep])
        params = []
        for algo in _all:
            idx = _all.index(algo)
            avg_reward[idx] = train(args, Gaussian_MAB, FUNCTION_MAP[algo])
            params.append(PARAMETER_MAP[algo])
        if args.plot:
            plot(avg_reward, _all, params)
    
    if args.runEpislon:
        epislons = [0, 0.01, 0.1, 0.5, 0.99]
        _all = ['e-Greedy'] * len(epislons)
        avg_reward = np.zeros([len(_all), args.max_timestep])
        params = []
        for idx in range(len(_all)):
            args.epislon = epislons[idx]
            avg_reward[idx] = train(args, Gaussian_MAB, FUNCTION_MAP[_all[idx]])
            params.append(('epislon', epislons[idx]))
        if args.plot:
            plot(avg_reward, _all, params)

    if args.runC:
        c = [0.1, 1, 2, 5, 10]
        _all = ['UCB'] * len(c)
        avg_reward = np.zeros([len(_all), args.max_timestep])
        params = []
        for idx in range(len(_all)):
            args.c = c[idx]
            avg_reward[idx] = train(args, Gaussian_MAB, FUNCTION_MAP[_all[idx]])
            params.append(('c', c[idx]))
        if args.plot:
            plot(avg_reward, _all, params)

    if args.runBandits:
        bandits = [5, 10, 50, 100, 500]
        _all = ['e-Greedy'] * len(bandits)
        avg_reward = np.zeros([len(_all), args.max_timestep])
        params = []
        for idx in range(len(_all)):
            args.num_of_bandits = bandits[idx]
            avg_reward[idx] = train(args, Gaussian_MAB, FUNCTION_MAP[_all[idx]])
            params.append(('bandits', bandits[idx]))
        if args.plot:
            plot(avg_reward, _all, params)