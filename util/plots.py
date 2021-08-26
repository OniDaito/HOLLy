"""   # noqa 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/           # noqa 
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/           # noqa 
Author : Benjamin Blundell - k1803390@kcl.ac.uk

plots.py - A number of plots of various stats
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cont_sigma(epochs, current_epoch: int, sigma: float, sigma_lookup: list, step_size: float) -> float:
    """
    If we are using _cont_sigma, we need to work out the linear
    relationship between the points. We call this each step.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    current_epoch : int
        The current epoch.
    sigma : float
        The current sigma.
    sigma_lookup : list
        The sigma lookup list of floats.

    Returns
    -------
    float
        The sigma to use
    """
    import math

    eps = epochs / (len(sigma_lookup) - 1)
    a = 0
    if current_epoch > 0:
        a = int(current_epoch / eps)
    b = a + 1
    assert b < len(sigma_lookup)
    assert a >= 0

    ssig = sigma_lookup[a]
    esig = sigma_lookup[b]

    steps = step_size * eps
    cont_factor = math.pow(float(esig) / float(ssig), 1.0 / float(steps))
    new_sigma = sigma * cont_factor
    return new_sigma

def plot_csv(sigmas, epochs=40, cont=False, title="Sigma", step_size=1):
    import math

    x = [i for i in range(epochs)]
    y = []

    if cont:
        sigma = sigmas[0]
        for i in range(epochs):
            sigma = cont_sigma(epochs, i, sigma, sigmas, step_size)
            y.append(sigma)
    else:
        s = float(len(sigmas)) / float(epochs)
        for i in range(epochs):
            idx = math.floor(i * s)
            y.append(sigmas[idx])

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Input Sigma')
    ax.set_title(title)
    ax.plot(x, y)
    plt.savefig('sigma.png')


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Render an image.")
   
    parser.add_argument(
        "--sigma-file",
        help="The sigma curve to plot",
    )
    parser.add_argument(
        "--title",
        help="The title of the plot",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=40,
        help="How many epochs?",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1.0,
        help="Step size?",
    )
    parser.add_argument(
        "--cont",
        default=False,
        action="store_true",
        help="Continuous Sigma?",
        required=False,
    )

    args = parser.parse_args()

    if os.path.isfile(args.sigma_file):
        with open(args.sigma_file, "r") as f:
            ss = f.read()
            sigma_lookup = []
            tokens = ss.replace("\n", "").split(",")
            for token in tokens:
                sigma_lookup.append(float(token))

        plot_csv(sigma_lookup, args.num_epochs, args.cont, args.title, args.step_size)