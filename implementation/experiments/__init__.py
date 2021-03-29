import argparse
from typing import Tuple
from dask.distributed import Client, LocalCluster
from dask_mpi import initialize
from distributed.scheduler import logger
import sys, signal, socket


def run_arg_parser() -> Tuple[int, int, int]:

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=98765, help='initial seed for rng')
    parser.add_argument('-j', '--jobs', type=int, default=-1, help='number or parallel workers')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs to run')
    args = parser.parse_args()
    return args.seed, args.jobs, args.epochs


def create_client(n_jobs: int) -> Client:

    if n_jobs == -1:
        initialize(interface='ib0')
        client = Client(timeout='180s')

        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        logger.info(f"Dashboard available at: {host}:{port}")

        return client
    else:
        cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1)
        return Client(cluster)


def create_exit_handler(client):

    def handler(signum, _):
        if __name__ == '__main__':
            client.close()
            print('Stopping experiment early...')
            sys.exit()

    signal.signal(signal.SIGINT, handler)
