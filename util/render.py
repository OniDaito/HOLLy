""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

render.py - quick render functions

"""
import torch
from net.renderer import Splat
from util.math import VecRotTen, VecRot, TransTen, Points, PointsTen, Trans, Mask


def render(p: Points, m: Mask, rot: VecRot, trans: Trans, sig: float, splat: Splat):
    """ Utility function for ease of testing."""
    mask = m.to_ten(device="cpu")
    xr = torch.tensor([rot.x], dtype=torch.float32, device="cpu")
    yr = torch.tensor([rot.y], dtype=torch.float32, device="cpu")
    zr = torch.tensor([rot.z], dtype=torch.float32, device="cpu")
    xt = torch.tensor([trans.x], dtype=torch.float32, device="cpu")
    yt = torch.tensor([trans.y], dtype=torch.float32, device="cpu")
    points = PointsTen(device="cpu")
    points.from_points(p)
    r = VecRotTen(xr, yr, zr)
    tt = TransTen(xt, yt)
    out = splat.render(points, r, tt, mask=mask, sigma=sig)

    return out


def render_better(p: Points, m: Mask, a: VecRot, t: Trans, sig: float, splat: Splat):
    """ Utility function for ease of testing."""
    mask = m.to_ten()
    points = PointsTen(device="cpu")
    points.from_points(p)
    r = a.to_ten()
    tt = t.to_ten()
    out = splat.render(points, r, tt, mask=mask, sigma=sig)

    return out
