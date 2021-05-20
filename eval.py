""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

eval.py - load a model, set to evaluation mode and run a few
examples and spit out some stats.

To load a trained network:
  python eval.py --savedir ../path/to/saved

"""

import torch
import torch.nn.functional as F
import math
import random
import argparse
import sys
import os
from stats import stats as S
from tqdm import tqdm
from pyquaternion import Quaternion
from net.renderer import Splat
from util.image import save_image
from util.plyobj import load_obj, save_obj, save_ply
from util.loadsave import load_checkpoint, load_model
from util.image import NormaliseTorch, NormaliseNull
from util.math import VecRotTen, VecRot, TransTen, PointsTen


def angle_eval(args, model, points, batch_size, device):
    """For every angle, save the in and out so we can assess where the
    network is failing."""
    xt = 0.0
    yt = 0.0
    num_angles = 36
    lerps = 10

    # pp = 1.0 / num_angles ** (1. / 3)
    xt = torch.tensor([xt], dtype=torch.float32, device=device)
    yt = torch.tensor([yt], dtype=torch.float32, device=device)

    # Generate random rotations then lerp between them
    rand_rots = []
    for i in range(num_angles):
        rand_rots.append(VecRot(0, 0, 0).random())

    normaliser = NormaliseNull()
    if args.normalise_basic:
        normaliser = NormaliseTorch()

    # Load some base points from an obj
    loaded_points = load_obj(objpath=args.obj)
    mask = []
    for _ in loaded_points:
        mask.append(1.0)
    mask = torch.tensor(mask, device=device)
    scaled_points = PointsTen(device=device).from_points(loaded_points)

    for i in tqdm(range(num_angles-1)):
        rot_s = rand_rots[i]
        rot_n = rand_rots[i+1]
        qrot_s = Quaternion(axis=rot_s.get_normalised(), radians=rot_s.get_angle())
        qrot_n = Quaternion(axis=rot_n.get_normalised(), radians=rot_n.get_angle())

        for j in range(lerps):
            idx = i * lerps + j
            qrot = Quaternion.slerp(qrot_s, qrot_n, amount=float(j) / float(lerps))

            fx = qrot.axis[0] * qrot.radians
            fy = qrot.axis[1] * qrot.radians
            fz = qrot.axis[2] * qrot.radians

            xv = torch.tensor([fx], dtype=torch.float32, device=device)
            yv = torch.tensor([fy], dtype=torch.float32, device=device)
            zv = torch.tensor([fz], dtype=torch.float32, device=device)
            r = VecRotTen(xv, yv, zv)
            t = TransTen(xt, yt)

            # Stats turn on
            S.write_immediate((fx, fy, fz), "eval_rot_in", 0, 0, idx)

            # Setup our splatting pipeline which is added to both dataloader
            # and our network as they use thTraine same settings
            splat = Splat(math.radians(90), 1.0, 1.0, 10.0, device=device)
            result = splat.render(scaled_points, r, t, mask, sigma=args.sigma)
            save_image(result, args.savedir + "/" + "eval_in_" + str(idx).zfill(3) + ".jpg")

            target = result.reshape(1, 128, 128)
            target = target.repeat(batch_size, 1, 1, 1)
            target = target.to(device)
            target = normaliser.normalise(target)

            # We use tpoints because otherwise we can't update points
            # and keep working out the gradient cos pytorch weirdness
            model.set_sigma(args.sigma)
            output = model.forward(target, points)
            output = normaliser.normalise(output.reshape(batch_size, 1, 128, 128))
            loss = F.l1_loss(output, target)

            output = torch.squeeze(output.cpu()[0])
            save_image(output, args.savedir + "/" + "eval_out_" + str(idx).zfill(3) + ".jpg")
            rots = model.get_rots()

            S.write_immediate(rots[0], "eval_rot_out", 0, 0, idx)
            S.write_immediate(loss, "eval_loss", 0, 0,  idx)


def basic_eval(args, model, points, batch_size, device):
    """ Our basic evaluation step. """
    xr = 0.0
    yr = 0.0
    zr = 0.0
    xt = 0.0
    yt = 0.0

    if args.rots:
        xr = float(math.radians(args.rots[0]))
        yr = float(math.radians(args.rots[1]))
        zr = float(math.radians(args.rots[2]))
    if args.trans:
        xt = float(args.trans[0])
        yt = float(args.trans[1])

    xr = torch.tensor([xr], dtype=torch.float32, device=device)
    yr = torch.tensor([yr], dtype=torch.float32, device=device)
    zr = torch.tensor([zr], dtype=torch.float32, device=device)
    xt = torch.tensor([xt], dtype=torch.float32, device=device)
    yt = torch.tensor([yt], dtype=torch.float32, device=device)

    r = VecRotTen(xr, yr, zr)
    t = TransTen(xt, yt)

    normaliser = NormaliseNull()
    if args.normalise_basic:
        normaliser = NormaliseTorch()

    # Setup our splatting pipeline which is added to both dataloader
    # and our network as they use thTraine same settings
    splat = Splat(math.radians(90), 1.0, 1.0, 10.0, device=device)
    loaded_points = load_obj(objpath=args.obj)

    mask = []
    for _ in loaded_points:
        mask.append(1.0)

    mask = torch.tensor(mask, device=device)
    scaled_points = PointsTen(device=device).from_points(loaded_points)
    result = splat.render(scaled_points, r, t, mask=mask, sigma=args.sigma)
    trans_points = splat.transform_points(scaled_points, r, t)
    save_image(result.clone().cpu(), args.savedir + "/" + "eval_single_in.jpg")

    target = result.reshape(1, 128, 128)
    target = target.repeat(batch_size, 1, 1, 1)
    target = target.to(device)
    target = normaliser.normalise(target)

    # We use tpoints because otherwise we can't update points
    # and keep working out the gradient cos pytorch weirdness
    model.set_sigma(args.sigma)
    output = model.forward(target, points)
    output = normaliser.normalise(output.reshape(batch_size, 1, 128, 128))
    loss = F.l1_loss(output, target)
    print("Loss :", loss)
    print("Rotations returned:", model.get_rots())
    output = torch.squeeze(output.cpu()[0])
    save_image(output, args.savedir + "/" + "eval_single_out.jpg")
    rots = model.get_rots()
    print("Rots / Trans detected: ", rots[0])

    # Now save the input points
    vertices = []
    for p in trans_points:
        vertices.append((float(p[0][0]), float(p[1][0]), float(p[2][0]), 1.0))

    save_obj(args.savedir + "/" + "eval_in.obj", vertices)
    save_ply(args.savedir + "/" + "eval_in.ply", vertices)

    # ... and the output points
    xr = torch.tensor([rots[0][0]], dtype=torch.float32, device=device)
    yr = torch.tensor([rots[0][1]], dtype=torch.float32, device=device)
    zr = torch.tensor([rots[0][2]], dtype=torch.float32, device=device)
    if not args.no_translate:
        xt = torch.tensor([rots[0][3]], dtype=torch.float32, device=device)
        yt = torch.tensor([rots[0][4]], dtype=torch.float32, device=device)
    r = VecRotTen(xr, yr, zr)
    t = TransTen(xt, yt)
    trans_points = splat.transform_points(points, r, t)
    vertices = []

    for p in trans_points:
        vertices.append((float(p[0][0]), float(p[1][0]), float(p[2][0]), 1.0))

    save_obj(args.savedir + "/" + "eval_out.obj", vertices)
    save_ply(args.savedir + "/" + "eval_out.ply", vertices)


def evaluate(args, device, animate=False):
    """ Begin our training routine on the selected device."""
    # Continue training or start anew
    # Declare the variables we absolutely need
    S.on(args.savedir)
    model = None
    points = None

    if os.path.isfile(args.savedir + "/" + args.savename):
        (model, points) = load_checkpoint(
            args.savedir, args.savename, device, evaluation=True
        )
        model = load_model(args.savedir + "/model.tar", device)
        model.to(device)
        print("Loaded model", model)
    else:
        print("Error - need to pass in a model")
        return

    batch_size = model._final.size()[0]
    print("Batch Size :", batch_size)

    model.eval()
    random.seed()
    if args.predict_sigma:
        model.predict_sigma = True
    basic_eval(args, model, points, batch_size, device)
    if animate:
        angle_eval(args, model, points, batch_size, device)
    S.close()


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Shaper Eval")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--predict-sigma",
        action="store_true",
        default=False,
        help="Does this model predict sigma",
    )
    parser.add_argument(
        "--normalise-basic",
        action="store_true",
        default=False,
        help="Use a basic normaliser",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--savedir", default="./save", help="The name for checkpoint save directory."
    )
    parser.add_argument(
        "--obj", default="teapot.obj", help="Path to the groundruth obj file"
    )
    parser.add_argument(
        "--sigma", default=1.25, type=float, help="The sigma value for this testing"
    )
    parser.add_argument(
        "--rots",
        metavar="R",
        type=float,
        nargs=3,
        help="Rotation around X, Y, Z axis in degrees.",
    )
    parser.add_argument(
        "--animate", action="store_true", default=False, help="Evaluate the angles."
    )
    parser.add_argument(
        "--trans", metavar="R", type=float, nargs=2, help="Translation on X, Y plane"
    )
    parser.add_argument(
        "--savename",
        default="checkpoint.pth.tar",
        help="The name for checkpoint save file.",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        default=False,
        help="Turn off translation (default: false)",
    )

    # Initial setup of PyTorch
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    random.seed(args.seed)
    evaluate(args, device, args.animate)
    print("Finished Evaluation")

    sys.exit(0)
