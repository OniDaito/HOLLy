""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

data.py - our test functions using python's unittest.
This file tests the data classes like loader and such.

"""

import unittest
import math
import torch
from data.loader import Loader
from data.sets import DataSet, SetType
from data.buffer import Buffer
from data.batcher import Batcher
from net.renderer import Splat
from util.image import NormaliseTorch
import torch.optim as optim
from torch.profiler import (
    profile,
    ProfilerActivity,
)
from net.net import Net
from util.image import save_image
from util.math import PointsTen
from util.plyobj import load_obj
from util.profile import get_memory_usage, count_parameters
from train.loss import calculate_loss


class Profile(unittest.TestCase):
    def test_train(self):
        device = torch.device("cuda")
        train_set_size = 4000
        image_height = 128
        image_width = 256
        batch_size = 32
        sigma = 5.0

        # Setup our splatting pipeline. We use two splats with the same
        # values because one never changes its points / mask so it sits on
        # the gpu whereas the dataloader splat reads in differing numbers of
        # points.
        image_size = (image_height, image_width)
        splat_in = Splat(
            math.radians(90), 1.0, 1.0, 10.0, device=device, size=image_size
        )
        splat_out = Splat(
            math.radians(90), 1.0, 1.0, 10.0, device=device, size=image_size
        )
        objpath = "./objs/bunny_large.obj"

        data_loader = Loader(size=train_set_size, objpath=objpath, sigma=sigma)

        set_train = DataSet(SetType.TRAIN, train_set_size, data_loader)
        buffer_train = Buffer(
            set_train, splat_in, buffer_size=batch_size, device=device
        )

        # Load the bunny model
        base_points = PointsTen(device=device)
        base_points.from_points(load_obj("./objs/bunny_large.obj"))

        # Create our main network
        model = Net(
            splat_out,
        ).to(device)

        model.train()

        # Print Stats for our model
        variables = []
        variables.append({"params": model.parameters()})
        optimiser = optim.AdamW(variables, lr=0.0004)
        print(get_memory_usage())
        print(count_parameters(model))

        print("Starting training new model")
        normaliser = NormaliseTorch()

        # We'd like a batch rather than a similar issue.
        batcher = Batcher(buffer_train, batch_size=batch_size)
        data_loader.set_sigma(sigma)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
        ) as prof:

            for step, ddata in enumerate(batcher):
                target = ddata.data
                optimiser.zero_grad()

                # Shape and normalise the input batch
                target_shaped = normaliser.normalise(
                    target.reshape(
                        batch_size,
                        1,
                        image_height,
                        image_width,
                    )
                )
                output = normaliser.normalise(model(target_shaped, base_points))
                final_out = output[0].squeeze()
                final_in = target_shaped[0].squeeze()

                save_image(
                    torch.sum(final_in, dim=0),
                    name="test_profile_in_" + str(step) + ".jpg",
                )
                save_image(
                    torch.sum(final_out, dim=0),
                    name="test_profile_out_" + str(step) + ".jpg",
                )

                loss = calculate_loss(target_shaped, output)
                loss.backward()
                optimiser.step()
                print("Step", step)
                print("Loss", loss)

            buffer_train.set.shuffle()

        prof.export_chrome_trace("trace.json")
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_memory_usage", row_limit=100
            )
        )
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=100
            )
        )
        print(
            prof.key_averages(group_by_stack_n=5).table(
                sort_by="self_cuda_time_total", row_limit=100
            )
        )
