import multiprocessing
import os
import json
import resource
import time
import subprocess
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import psutil
import pandas as pd

from Scripts.MemoryMonitor import MemoryMonitor

PATH = './'
ROOD_DIR = os.path.abspath(os.getcwd())

def run_mytool(model, seed, approch, path):
    os.chdir(path)
    run_dl_command = "python run.py --model {} --seed {} --approach {}".format(model, seed, approch)
    print(run_dl_command)
    return subprocess.Popen(run_dl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main(target_path):
    mydata = pd.DataFrame()
    approch = ['tarantula',
               'ochiai',
               'dstar']
    model = ['lenet5',
             'vgg16',
             'resnet20']
    script_path = "../Code/DeepFault_old"
    model_name = []
    seed_num = []
    approch_name = []
    ex_time = []
    cpu_use = []
    memory_use = []
    time_use = []


    # Calcul Memory usage
    for s in range(10):
        for a in approch:
            for m in model:
                tracemalloc.start()
                start = time.time()

                process1 = multiprocessing.Process(target=run_mytool(m, s, a, script_path))



                ps = psutil.Process(process1.pid)
                cpu = ps.cpu_times()
                mem = ps.memory_percent()

                final_time = (time.time() - start)
                current, peak = tracemalloc.get_traced_memory()

                print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB; Time is {final_time}; CPU is {cpu}; Memory percentage is {mem}%")
                tracemalloc.stop()

                os.chdir(target_path)


if __name__ == '__main__':
    main(ROOD_DIR)
