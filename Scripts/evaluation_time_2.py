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

def evaluate_mytool(processId):
    ps = psutil.Process(processId)
    time_usage = ps.cpu_times().system
    cpu_usage = ps.cpu_percent(interval=None)
    memory_usage = ps.memory_percent()

    print(f"Time = {time_usage}, CPU = {cpu_usage}, Memory = {memory_usage}")

def run_dltools(model, seed, approch, path):
    os.chdir(path)
    run_dl_command = "python run.py --model {} --seed {} --approach {}".format(model, seed, approch)
    print(run_dl_command)
    p = subprocess.Popen(run_dl_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    PID = p.pid
    print(PID)
    ps = psutil.Process(PID)
    time_usage = ps.cpu_times().system
    cpu_usage = ps.cpu_percent(interval=5)
    memory_usage = ps.memory_percent()
    output = ""
    for line in p.stdout.readlines():
        output = output + str(line) + '\n'
    retval = p.wait()
    if retval == 0:
        print("program run successfully!")
    else:
        print("Error in running program!")
        print(output)
    return cpu_usage, memory_usage, time_usage


def get_file(root_dir):
    file_set = []
    for dir_, _, files in os.walk(root_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[-1].lower()
            if ext == ".json":
                rel_file = os.path.join(dir_, file_name)
                file_set.append(rel_file)
    # print(file_set)
    return file_set


def read_Json(path):
    json_list = []
    with open(path) as json_file:
        jsondict = json.load(json_file)
    # print(jsondict)
    return jsondict


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





    # for i in range(1):
    #     for x in approch:
    #         for y in model:
    #             start = time.time()
    #             cpu, memory, time_ = run_dltools(y, i, x, script_path)
    #             final_time = (time.time() - start)
    #             model_name.append(y)
    #             approch_name.append(x)
    #             seed_num.append(i)
    #             ex_time.append(final_time)
    #             cpu_use.append(cpu)
    #             memory_use.append(memory)
    #             time_use.append(time_)
    #             print("Execution information taken to run Deepfault, times : " + str(
    #                 final_time) + " Seconds, cpu : " + str(
    #                 cpu) +
    #                   "% memory : " + str(memory) +
    #                   "%")
    # mydata["model"] = model_name
    # mydata["seed"] = seed_num
    # mydata["approch"] = approch_name
    # mydata["Time"] = ex_time
    # mydata["cpu"] = cpu_use
    # mydata["memory"] = memory_use
    # mydata.to_csv(target_path + '/my_final_data.csv', mode='w', index=False, header=True)
    # for command in list_command:
    #     run_dltools(command["command"], script_path)
    # final_time = (time.time() - start)


if __name__ == '__main__':
    main(ROOD_DIR)
