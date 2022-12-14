# Flower + LEAF

This repository provides an example of federated learning with [Flower framework](https://github.com/adap/flower) on [LEAF](https://github.com/TalwalkarLab/leaf). 


## Installation

```bash
$ git clone --recursive https://github.com/yonetaniryo/flwr_leaf.git
$ cd flwr_leaf
$ pip install .[dev]
```

## Generate Synthetic Data
See https://github.com/TalwalkarLab/leaf/tree/master/data/synthetic for input arguments.

```bash
$ cd leaf/data/synthetic
$ python main.py -num-tasks 1000 -num-classes 5 -num-dim 64     # `num_tasks`: maximum number of clients
$ sh preprocess.sh -s niid --sf 1.0 -k 100 -t sample --tf 0.6
```


## Run FedAvg on Synthetic Data
```bash
$ cd scripts
$ bash run.sh
```