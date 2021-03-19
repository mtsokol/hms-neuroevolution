import argparse
from typing import Tuple
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import os, sys, signal


def run_arg_parser() -> Tuple[int, int, int]:

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=98765, help='initial seed for rng')
    parser.add_argument('-j', '--jobs', type=int, default=9, help='number or parallel workers')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs to run')
    args = parser.parse_args()
    return args.seed, args.jobs, args.epochs


def create_client(n_jobs: int) -> Client:

    if 'PARTITION' in os.environ and 'GRANT' in os.environ:
        cluster = SLURMCluster(queue=os.environ['PARTITION'],
                               project=os.environ['GRANT'],
                               cores=8,
                               processes=8,
                               memory='16 GB',
                               walltime='00:20:00',
                               interface='ib0',  # interface for the workers
                               scheduler_options={'interface': 'eth55'})  # interface for the scheduler
        cluster.scale(jobs=n_jobs)
        return Client(cluster)
    else:
        cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1)
        return Client(cluster)


def create_exit_handler(future, client):

    def handler(signum, _):
        if __name__ == '__main__':
            client.cancel([future])
            print('Stopping experiment early. Saving scores...')
            sys.exit()

    signal.signal(signal.SIGINT, handler)
