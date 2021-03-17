from datetime import datetime
import os


def make_dir():

    dir_name = datetime.now().strftime('experiment-%Y-%m-%d--%H-%M')

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    return dir_name
