"""
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

 net.py - an attempt to find the 3D shape from an image.

"""
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from net.renderer import Splat
from util.math import VecRotTen, TransTen, PointsTen


def conv_size(x, padding=0, dilation=0, kernel_size=5, stride=1) -> int:
    """
    Return the size of the convolution layer given a set of parameters

    Parameters
    ----------
    x : int
        The size of the input tensor

    padding: int
        The conv layer padding - default 0
    
    dilation: int
        The conv layer dilation - default 0
    
    kernel_size: int
        The conv layer kernel size - default 5

    stride: int 
        The conv stride - default 1

    """
    return int(((x + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)


def num_flat_features(x):
    """
    Return the number of features of this neural net layer,
    if it were flattened.

    Parameters
    ----------
    x : torch.Tensor
        The layer in question.

    Returns
    -------
    int
        The number of features

    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class Net(nn.Module):
    """The defininition of our convolutional net that reads our images
    and spits out some angles and attempts to figure out the loss
    between the output and the original simulated image.
    """

    def __init__(self, splat: Splat, predict_translate=False, predict_sigma=False, max_trans=0.1):
        """
        Initialise the model.

        Parameters
        ----------
        splat : Splat
            The renderer splat.
        predict_translate : bool
            Do we predict the translation (default: False).
        predict_sigma : bool
            Do we predict the sigma as well (default: False).
        max_trans : float
            The scalar attached to the translation (default: 0.1).

        Returns
        -------
        nn.Module
            The model itself

        """
        super(Net, self).__init__()
        # Conv layers
        self.batch1 = nn.BatchNorm2d(16)
        self.batch2 = nn.BatchNorm2d(32)
        self.batch2b = nn.BatchNorm2d(32)
        self.batch3 = nn.BatchNorm2d(64)
        self.batch3b = nn.BatchNorm2d(64)
        self.batch4 = nn.BatchNorm2d(128)
        self.batch4b = nn.BatchNorm2d(128)
        self.batch5 = nn.BatchNorm2d(256)
        self.batch5b = nn.BatchNorm2d(256)
        self.batch6 = nn.BatchNorm2d(256)

        # Model the sigma
        self.predict_sigma = predict_sigma

        # Model translation
        self.predict_translate = predict_translate

        # Added more conf layers as we aren't using maxpooling
        # TODO - we only have one pseudo-maxpool at the end
        # TODO - do we fancy some drop-out afterall?
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)
        csize = conv_size(splat.size[0], padding=1, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        csize = conv_size(csize, padding=1, stride=2, kernel_size=3)

        self.conv2b = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        csize = conv_size(csize, padding=1, stride=1, kernel_size=3)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        csize = conv_size(csize, padding=1, stride=2, kernel_size=3)

        self.conv3b = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        csize = conv_size(csize, padding=1, stride=1, kernel_size=3)

        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        csize = conv_size(csize, padding=1, stride=2, kernel_size=3)

        self.conv4b = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        csize = conv_size(csize, padding=1, stride=1, kernel_size=3)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        csize = conv_size(csize, padding=1, stride=2, kernel_size=3)

        self.conv5b = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        csize = conv_size(csize, padding=1, stride=1, kernel_size=3)

        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        csize = conv_size(csize, padding=1, stride=2, kernel_size=3)

        # Fully connected layers
        #self.fc1 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(csize, 512)
        nx = 3

        if self.predict_translate:
            nx += 2
        if self.predict_sigma:
            nx += 1

        self.fc2 = nn.Linear(512, nx)

        self.max_shift = max_trans
        self.splat = splat
        self.device = self.splat.device
        self._lidx = 0
        self._mask = torch.zeros(splat.size, dtype=torch.float32)

        self.layers = [
            self.conv1,
            self.conv2,
            self.conv2b,
            self.conv3,
            self.conv3b,
            self.conv4,
            self.conv4b,
            self.conv5,
            self.conv5b,
            self.conv6,
            self.fc1,
            self.fc2,
        ]

        # Specific weight and bias initialisation
        for layer in self.layers:
            torch.nn.init.xavier_uniform(layer.weight)
            layer.bias.data.fill_(random.random() * 0.001)

    def __iter__(self):
        return iter(self.layers)

    def __next__(self):
        if self._lidx > len(self.layers):
            self._lidx = 0
            raise StopIteration
        rval = self.layers[self._lidx]
        self._lidx += 1
        return rval

    def to(self, device):
        super(Net, self).to(device)
        self.splat.to(device)
        self.device = device
        return self

    def set_splat(self, splat: Splat):
        self.splat = splat

    def set_sigma(self, sigma: float):
        """
        Set the total sigma we are rendering. Our network will
        predict a bit more based on the wobble.

        Parameters
        ----------
        sigma : float
            The sigma value used by the renderer

        Returns
        -------
        None

        """

        self.sigma = sigma

    def get_rots(self):
        return self._final

    def forward(self, target: torch.Tensor, points: PointsTen):
        """
        Our forward pass. We take the input image (x), the
        vector of points (x,y,z,w) and run it through the model. Offsets
        is an optional list the same size as points.

        Initialise the model.

        Parameters
        ----------
        target : torch.Tensor
            The target image, as a tensor.
        points : PointsTen
            The points we are predicting.

        Returns
        -------
        None

        """
        x = F.leaky_relu(self.batch1(self.conv1(target)))

        x = F.leaky_relu(self.batch2(self.conv2(x)))
        x = F.leaky_relu(self.batch2b(self.conv2b(x)))

        x = F.leaky_relu(self.batch3(self.conv3(x)))
        x = F.leaky_relu(self.batch3b(self.conv3b(x)))

        x = F.leaky_relu(self.batch4(self.conv4(x)))
        x = F.leaky_relu(self.batch4b(self.conv4b(x)))

        x = F.leaky_relu(self.batch5(self.conv5(x)))
        x = F.leaky_relu(self.batch5b(self.conv5b(x)))

        x = F.leaky_relu(self.batch6(self.conv6(x)))
        x = x.view(-1, num_flat_features(x))

        x = F.leaky_relu(self.fc1(x))
        self._final = self.fc2(x)  # Save this layer for later use

        # Check that self._mask is not none
        # TODO - also check that it is the correct shape (actually is
        # recreation faster than the if check?)
        # if self._mask is None:
        self._mask = points.data.new_full([points.data.shape[0], 1, 1], fill_value=1.0)

        # Now take the rotations here and splat them through our last
        # process
        # TODO - this process is the same in dataloader so we should
        # probably share the code better
        images = []

        for idx, rot in enumerate(self._final):
            tx = torch.tensor([0.0], dtype=torch.float32, device=self.device)
            ty = torch.tensor([0.0], dtype=torch.float32, device=self.device)
            nx = 2

            if self.predict_translate:
                nx += 2
                tx = (torch.tanh(rot[3]) * 2.0) * self.max_shift
                ty = (torch.tanh(rot[4]) * 2.0) * self.max_shift

            sp = nn.Softplus()
            final_sigma = self.sigma

            if self.predict_sigma:
                nx += 1
                final_sigma = sp(rot[nx])
            else:
                final_sigma = self.sigma
            # exp(tanh(s) * log(3))

            r = VecRotTen(rot[0], rot[1], rot[2])
            t = TransTen(tx, ty)

            images.append(
                self.splat.render(points, r, t, self._mask, final_sigma).reshape(
                    (1, self.splat.size[0], self.splat.size[1])
                )
            )
        # TODO - should we return the params we've predicted as well?
        return torch.stack(images)


# Our drawing graph functions. We rely / have borrowed from the following
# python libraries:
# https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py
# https://github.com/willmcgugan/rich
# https://graphviz.readthedocs.io/en/stable/


def draw_graph(start, watch=[]):
    """
    Draw our graph.

    Parameters
    ----------
    start : torch.Tensor
        Where do we begin to draw our loss? Typically, the loss.
    watch : list
        The list of tensors to watch, if any.

    Returns
    -------
    None

    """

    from graphviz import Digraph

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
    )
    graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    assert hasattr(start, "grad_fn")
    if start.grad_fn is not None:
        _draw_graph(start.grad_fn, graph, watch=watch)

    size_per_element = 0.15
    min_size = 12

    # Get the approximate number of nodes and edges
    num_rows = len(graph.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    graph.graph_attr.update(size=size_str)
    graph.render(filename="net_graph.jpg")


def _draw_graph(var, graph, watch=[], seen=[], indent="", pobj=None):
    """
    Internal recursive function.
    """
    from rich import print

    if hasattr(var, "next_functions"):
        for fun in var.next_functions:
            joy = fun[0]
            if joy is not None:
                if joy not in seen:
                    label = (
                        str(type(joy))
                        .replace("class", "")
                        .replace("'", "")
                        .replace(" ", "")
                    )
                    label_graph = label
                    colour_graph = ""
                    seen.append(joy)

                    if hasattr(joy, "variable"):
                        happy = joy.variable
                        if happy.is_leaf:
                            label += " \U0001F343"
                            colour_graph = "green"

                            for (name, obj) in watch:
                                if obj is happy:
                                    label += (
                                        " \U000023E9 "
                                        + "[b][u][color=#FF00FF]"
                                        + name
                                        + "[/color][/u][/b]"
                                    )
                                    label_graph += name
                                    colour_graph = "blue"
                                    break

                            vv = [str(obj.shape[x]) for x in range(len(obj.shape))]
                            label += " [["
                            label += ", ".join(vv)
                            label += "]]"
                            label += " " + str(happy.var())

                    graph.node(str(joy), label_graph, fillcolor=colour_graph)
                    print(indent + label)
                    _draw_graph(joy, graph, watch, seen, indent + ".", joy)
                    if pobj is not None:
                        graph.edge(str(pobj), str(joy))
