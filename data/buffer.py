""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

databuffer.py - a class that takes a dataset and buffers it for us. Usually
putting partial bit on the GPU but also CPU.

The CPU based buffer will render as many of the images into main memory as it
can from the dataset. The GPU renderer builds on top of this, by pinning that 
memory and then putting it into the GPU before batching.

Buffer                  ---> Set            --->  Loader
Fill up and render           [ids reserved]       Hold data for rendering


"""

import torch
from astropy.io import fits
from tqdm import tqdm
from data.sets import DataSet
from data.loader import ItemType
from util.math import PointsTen, VecRotTen, TransTen
from scipy.ndimage import gaussian_filter


class BufferItem(object):
    """
    An item that lives in the buffer. A Datum. This is
    at the very least, an image tensor.

    """

    def __init__(self, rendered):
        self.datum = rendered

    def flatten(self):
        return self.datum


class ItemRendered(BufferItem):
    """
    A rendered Datum. Given some basic parameters, group
    these parameters with the rendereed image.
    Useful for all the type checking we would like to do.
    """

    def __init__(
        self,
        rendered: torch.Tensor,
        rotation: VecRotTen,
        translation: TransTen,
        sigma: torch.Tensor,
    ):
        super().__init__(rendered)
        self.rotation = rotation
        self.translation = translation
        self.sigma = sigma

    def flatten(self):
        return (self.datum, self.rotation, self.translation, self.sigma)


class BaseBuffer(object):
    def __init__(self, dataset: DataSet, buffer_size=1000, device=torch.device("cpu")):
        """
        Create our BaseBuffer

        Parameters
        ----------
        dataset : Dataset
            The dataset behind the buffer
        buffer_size : int
            How big is our buffer (default)
        device : str
            The device the buffer lives on - CUDA/cpu (default: "cpu)

        Returns
        -------
        BaseBuffer
        """
        # TODO - could be a set OR another buffer - think cpu/gpu buffering
        self.device = device
        self.set = dataset
        self.buffer = []
        self.buffer_size = buffer_size
        self.counter = 0

    def reset(self):
        """
        Reset the buffer. Clear it, and set the counter to 0.

        Parameters
        ----------
        None

        Returns
        -------
        self : BaseBuffer
        """
        self.counter = 0
        del self.buffer[:]
        return self

    def __getitem__(self, idx) -> BufferItem:
        """
        Get an item out of our buffer

        Parameters
        ----------
        idx : int
            Index of the item to return
        Returns
        -------
        tuple
        """
        try:
            return self.buffer[idx]
        except Exception as e:
            print("Buffer Exception", e)
            raise e

    def __iter__(self):
        return self

    def __len__(self):
        """
        Return the length of the underlying set, not the buffer.

        Parameters
        ----------
        None

        Returns
        -------
        int
        """
        return len(self.set)

    def fill(self):
        """Placeholder for now."""
        assert False

    def __next__(self) -> BufferItem:
        """Return the rendered image, the transform list,
        and the sigma. or just rendered image if we are going for FITS
        image."""

        try:
            if self.counter >= len(self.buffer):
                if self.set.remaining() > 0:
                    self.fill()
                else:
                    self.set.reset()
                    self.reset()
                    raise StopIteration("Reached the end of the dataset.")

            datum = self.buffer[self.counter]
            self.counter += 1
            return datum
        except StopIteration:
            self.counter = 0
            raise StopIteration("Reached the end of the dataset.")
        except IndexError:
            self.counter = 0
            raise StopIteration("Reached the end of the dataset.")
        except Exception as e:
            print("Base Buffer Exception", e)
            raise e


class Buffer(BaseBuffer):
    """
    A Buffer holds the actual data, either rendered and ready to go or it
    doesn't have a renderer and just holds the raw data. Renderer is either
    our Splat class or the fits image loader.
    """

    def __init__(
        self, dataset: DataSet, renderer, buffer_size=1000, device=torch.device("cpu")
    ):

        # TODO - could be a set OR another buffer - think cpu/gpu buffering
        super().__init__(dataset, buffer_size, device)
        self.renderer = renderer

    def fill(self):
        """
        Perform a fill. We go all the way through the dataset and into the
        DataLoader, rendering enough to fill the buffer.

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        # Using try catch as although we get another tqdm print, we get an
        # elegant reload ready for the next epoch
        self.counter = 0
        try:
            del self.buffer[:]

            for i in range(0, min(self.buffer_size, self.set.remaining())):
                # Here is where we render and place into the buffer
                datum = self.set.__next__()
                if self.renderer is not None:
                    assert datum.type == ItemType.SIMULATED
                    points = PointsTen(device=self.device)
                    points.from_points(datum.points)
                    mask = datum.mask.to_ten(device=self.device)
                    r = datum.angle_axis.to_ten(device=self.device)
                    t = datum.trans.to_ten(device=self.device)
                    rendered = self.renderer.render(
                        points, r, t, mask=mask, sigma=datum.sigma
                    )

                    self.buffer.append(ItemRendered(rendered, r, t, datum.sigma))

        except Exception as e:
            print("Buffer exception on Fill", e)
            raise e

        return self

    def image_size(self):
        """
        Return the image size of the renderer in this buffer.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
        """
        return self.renderer.size


class BufferImage(BaseBuffer):
    """
    This buffer requires no splat as it loads images instead of
    rendering from an obj.
    """

    def __init__(
        self,
        dataset,
        image_size=(128, 128),
        buffer_size=1000,
        device=torch.device("cpu"),
        blur=True,

    ):
        """
        Build our BufferImage - a buffer that loads images as oppose to
        rendering them using the Splat class.

        Parameters
        ----------
        dataset : Dataset
            The dataset behind this buffer.
        image_size : tuple
            The size of the images to be expected - default: (128, 128)
        buffer_size : int
            Default 1000
        device : str
            The device to bind the buffer to (CUDA/cpu) - default: "cpu"

        Returns
        -------
        self
        """
        super().__init__(dataset, buffer_size, device)
        self.image_dim = image_size
        self.blur = blur

    def fill(self):
        """
        Perform a fill. We go all the way through the dataset and into the
        DataLoader, rendering enough to fill the buffer.

        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        # TODO - if this is a CPU buffer, pin the memory

        self.counter = 0

        try:
            del self.buffer[:]

            for _ in tqdm(
                range(0, min(self.buffer_size, self.set.remaining())),
                desc="Filling Image buffer",
            ):
                # Here is where we render and place into the buffer
                datum = self.set.__next__()
                assert datum.type == ItemType.FITSIMAGE

                with fits.open(datum.path) as w:
                    hdul = w[0].data.byteswap().newbyteorder().astype('float32')
                    timg = torch.tensor(hdul, dtype=torch.float32, device=self.device)
                    if len(timg.shape) == 3:
                        timg = torch.sum(timg, 0)

                    if self.blur and datum.sigma > 1.0:
                        timg = gaussian_filter(timg.cpu(), sigma=datum.sigma)
                        timg = torch.tensor(timg, device=self.device)

                    item = BufferItem(timg)
                    self.buffer.append(item)

        except Exception as e:
            print("Buffer exception", e)
            raise e

    def image_size(self):
        """The renderer is what holds the final image size."""
        return self.image_dim
