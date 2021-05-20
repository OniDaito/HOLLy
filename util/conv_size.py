"""   # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/           # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/           # noqa 
Author : Benjamin Blundell - k1803390@kcl.ac.uk

conv_size.py - A short function that works out the new size
of a conv layer, given things like dilation and stride
"""


import argparse

lin = 128
padding = 1
dilation = 1
kernel_size = 5
stride = 1

parser = argparse.ArgumentParser(description="Conv layer size")
parser.add_argument("--ins", type=int, default=128)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--padding", type=int, default=1)
parser.add_argument("--dilation", type=int, default=1)
parser.add_argument("--kernel", type=int, default=5)

args = parser.parse_args()

lin = args.ins
stride = args.stride
dilation = args.dilation
kernel_size = args.kernel
stride = args.stride
padding = args.padding

lout = ((lin + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1

print("Output", lout)
