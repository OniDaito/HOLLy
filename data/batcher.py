""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

batcher.py - tyrns single datum access into a set of data,
also known as a batch. This sits atop something that
supports iteration, typically a buffer.

"""

import torch
from data.buffer import BufferItem, ItemRendered


class Batch(object):
    ''' A little dictionary of sorts that holds the actual data we need 
    for the neural net (the images) and the associated data used to make
    these images.'''

    def __init__(self, batch_size: int, isize, device):
        self._idx = 0

        self.data = torch.zeros(
            (batch_size, 1, isize[0], isize[1]),
            device=device,
        )

        self.rotations = []
        self.translations = []
        self.sigmas = []
        self.stretches = []

    def add_datum(self, datum: BufferItem):
        self.data[self._idx][0] = datum.datum

        if isinstance(datum, ItemRendered):
            self.rotations.append(datum.rotation)
            self.translations.append(datum.translation)
            self.sigmas.append(datum.sigma)

        self._idx += 1


class Batcher:
    def __init__(self, buffer, batch_size=16):
        """
        Create our batcher

        Parameters
        ----------
        buffer : Buffer
            The buffer behind the batcher
        batch_size : int
            How big is the batch?

        Returns
        -------
        Batcher
        """
        self.batch_size = batch_size
        self.buffer = buffer
        self.device = buffer.device
        self.isize = self.buffer.image_size()

    def __iter__(self):
        return self

    def __len__(self):
        """ Return the number of batches."""
        return int(len(self.buffer) / self.batch_size)

    def __next__(self) -> BufferItem:
        """ Return the 'next' BufferItem in this buffer."""
        batch = torch.zeros(
            (self.batch_size, 1, self.isize[0], self.isize[1]),
            dtype=torch.float32,
            device=self.device,
        )

        batch = Batch(self.batch_size, self.buffer.image_size(), self.buffer.device)

        try:
            for _ in range(self.batch_size):
                datum = self.buffer.__next__()
                batch.add_datum(datum)

            return batch

        except StopIteration:
            raise StopIteration("Batcher reached the end of the dataset.")

        except Exception as e:
            print("Batcher Exception", e)
            raise e
