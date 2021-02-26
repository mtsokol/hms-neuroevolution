import argparse
from typing import Tuple


def run_arg_parser() -> Tuple[int, int, int]:

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=98765, help='initial seed for rng')
    parser.add_argument('-j', '--jobs', type=int, default=8, help='number or parallel workers')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs to run')
    args = parser.parse_args()
    return args.seed, args.jobs, args.epochs
