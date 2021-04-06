# HMS Neuroevolution

This repository is an implementation of master's thesis project: "Application of the hierarchic memetic strategy HMS in neuroevolution"

Here we examine the [HMS framework](https://www.sciencedirect.com/science/article/abs/pii/S1877750318307233) in the setting of neuroevolution. 
Following the findings from [Uber AI Labs on neuroevolution](https://arxiv.org/abs/1712.06567) 
we extend the notion of genetic algorithm to a hierarchic structure of such algorithms.
Agents were evaluated on selected Atari games and simple control systems available via Gym library.

Project supports parallel and distributed execution targeted at HPC clusters.


## Overview

Examined RL environments are: `CartPole`, `LunarLander` and Atari games such as `Frostbite`.

Project provides two types of genotype encoding:

- Fixed length

- Variable length - as described in [Uber AI Labs paper](https://arxiv.org/abs/1712.06567), requires access to the noise table.


## Installation

Required dependencies are available in `requirements.txt`.


## Usage

Test `CartPole` experiment can be executed via:

```shell
python3 -m implementation.experiments.hms_cartpole_sea -j 8
```

The output placed in a separate directory provides:

- human-readable experiment description
- snapshots of elite models within `checkpoints` directory
- scores of each evaluated deme (used for histograms)
- output of experiment process in `scores.txt`
- experiment evaluation plot

Available arguments for execution are:

```
usage: hms_cartpole_sea.py [-h] [-s SEED] [-j JOBS] [-e EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  initial seed for rng
  -j JOBS, --jobs JOBS  number or parallel workers, if -1 then it uses dask-mpi
  -e EPOCHS, --epochs EPOCHS number of epochs to run
```


### Atari

Variable length genotype encoding, mostly used in Atari environment, requires noise table for drawing samples. 
It can be created via:

```shell
python3 scripts/create_noise_table.py
```


### Executing on HPC cluster

HPC execution is provided via `dask-mpi` library. Exemplary script file is placed at `scripts/dask-mpi-batch.sh`.
It's adjusted to be used on [PLGrid Prometheus supercomputer](https://kdm.cyfronet.pl/portal/Prometheus:en) and can be run via:

```shell
sbatch scripts/dask-mpi-batch.sh
```


### Implementing your own experiment & environment

Custom experiments and environments can be easily created by implementing proper abstract class. 

By implementing `implementation/experiments/base_experiment.py` one can express what an individual is and how to evaluate it.

Experiment script should follow e.g. `implementation/experiments/hms_cartpole_sea.py`.

Environment base class (`implementation/environment/base_env.py`) imposes `Gym` usage and generally covers most of its environments.


### Visualization

Project provides a few notebooks for episode recording for elites (fun to watch!) and 
plotting (in `/notebooks` directory - detailed instructions inside a notebook). 

Please see live example evaluation here: https://youtu.be/t3P0w0I7Xw8
