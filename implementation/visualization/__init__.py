from datetime import datetime
import os

NOW = datetime.now()
DIR = NOW.strftime('experiment-%Y-%m-%d--%H-%M')

if not os.path.exists(DIR):
    os.makedirs(DIR)
