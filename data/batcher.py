""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

batcher.py - tyrns single datum access into a set of data,
also known as a batch. This sits atop something that
supports iteration, typically a buffer.

"""

import torch


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

    def __next__(self) -> tuple:
        batch = torch.zeros(
            (self.batch_size, 1, self.isize[0], self.isize[1]),
            dtype=torch.float32,
            device=self.device,
        )
        rotations = []
        sigmas = []
        translations = []

        try:
            for i in range(self.batch_size):
                datum = self.buffer.__next__()
                batch[i][0] = datum[0]
                if len(datum) == 4:
                    rotations.append(datum[1])
                    translations.append(datum[2])
                    sigmas.append(datum[3])

            if len(rotations) > 0:
                return (batch, rotations, translations, sigmas)
            return (batch,)

        except StopIteration:
            raise StopIteration("Batcher reached the end of the dataset.")

        except Exception as e:
            raise e
