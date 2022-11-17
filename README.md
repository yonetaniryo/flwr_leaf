# Flower + LEAF


## Synthetic dataset

## Data generation
```bash
$ python main.py -num-tasks 1000 -num-classes 5 -num-dim 64     # `num_tasks`: number of clients
$ ./preprocess.sh -s niid --sf 1.0 -k 100 -t sample --tf 0.6
```