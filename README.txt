To run a single algorithm, type:
python main.py --algo [algo name]

You can specify the parameters. e.g.
python main.py --algo e-Greedy --epislon 0.01

To run three algorithms at the same time, type:
python main.py --runAll

To run e-Greedy with different parameters epsilon at the same time, type:
python main.py --runEpislon

To run UCB with different parameters c at the same time, type:
python main.py --runC

To run e-Greedy with different number of bandits at the same time, type:
python main.py --runBandits

If you want to plot the result, add --plot at the end of every command.