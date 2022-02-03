""" # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa 
Author : Benjamin Blundell - k1803390@kcl.ac.uk

profile.py - a few functions that help us do some 
profiling.

"""

import torch
from prettytable import PrettyTable
from pynvml.smi import nvidia_smi

nvsmi = nvidia_smi.getInstance()


def get_memory_usage():
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    return "%d %s" % (usage["used"], usage["unit"])


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
