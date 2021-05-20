""" check.py - Small check function to detect problems with our NaN 
and similar.

"""
import colorama
import torch
import pdb
import traceback
from colorama import Fore, Back, Style
from torch import autograd

colorama.init()


class GuruMeditation(autograd.detect_anomaly):
    def __init__(self):
        super(GuruMeditation, self).__init__()

    def __enter__(self):
        super(GuruMeditation, self).__enter__()
        return self

    def __exit__(self, type, value, trace):
        super(GuruMeditation, self).__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            halt(str(value))


def halt(msg):
    print(Fore.RED + "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print(Fore.RED + "┃ Software Failure. Press left mouse button to continue ┃")
    print(Fore.RED + "┃       Guru Meditation '00000004, 0000AAC0             ┃")
    print(Fore.RED + "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
    print(Style.RESET_ALL)
    print(msg)
    pdb.set_trace()
