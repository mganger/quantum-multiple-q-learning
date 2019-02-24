# Quantum Multiple Q-learning

## Overview
This repository holds reference implementations of the algorithms described in
[Quantum Multiple Q-learning](https://doi.org/10.4236/ijis.2019.91001) [1].

## Compiling and Running

The only requirements are a recent version of `g++`. This has only been tested
in Linux environments.  To compile binaries for each algorithm, run `make` in
the `src` directory. The binaries are placed in `src/bin`. It takes arguments

	<algo> <parameter_file> <environment_file> [<n_iter> <n_steps_convergence>]

For example,

	bin/qqrl params/basic environments/simple 100

This will run 1000 episodes of `qqrl` using the basic parameters and simple
environment, averaging over 100 episodes (if `n_iter` is not provided, it will
assume a value of `1`). The output columns are in the format

	episode branch branch_count average_number_steps average_reward average_state_values...

If the argument `n_steps_convergence` is provided, the policy is initialized
and updated until it takes `n_steps_convergence` timesteps for the agent to
reach the final state by it.

For diagnostic information about parameters and the environment, squelch
`stdout` and view the output of `stderr`, e.g.

	bin/qqrl params/basic environments/simple 100 > /dev/null

## References
[1] M. Ganger and W. Hu, “Quantum Multiple Q-Learning,” International Journal
    of Intelligence Science, vol. 09, no. 01, pp. 1–22, 2019.
