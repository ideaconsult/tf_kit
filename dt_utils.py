#
# General Data handling utils, like file and np array iterators.
#
# Copyright (C) 2017, IDEAConsult Ltd.
#
# Author: Ivan (Jonan) Georgiev

"""
@@ArrayBatchIterator
@@FileBatchIterator
"""

import numpy as np
import random as rnd
import os
from .nn_utils import packed_images_reader
from .dt_constants import *


class ArrayBatchIterator:
    """np.array batch iterator"""
    def __init__(self, data,
                 batch_size=100,
                 allow_smaller_batch=False,
                 shuffle=False,
                 num_epochs=None):
        """
        Iterate on a numpy array, shuffling it, if needed
        :param data: The data as np.array()
        :param batch_size: The batch size to strive to deliver.
        :param allow_smaller_batch: Whether the last batch can be smaller.
        :param shuffle: Whether to shuffle data.
        :param num_epochs: Number of epochs of data iteration.
        """
        self._num_epochs = num_epochs if num_epochs is not None else MAX_EPOCHS
        self._shuffle = shuffle
        self._data = data
        self._batch_size = batch_size
        self._data_len = len(data)
        self._num_batches = int((self._data_len + batch_size - 1) / batch_size)
        self._allow_smaller = allow_smaller_batch
        self.__iter__()

    def __iter__(self):
        self._epoch = 0
        if self._shuffle:
            np.random.shuffle(self._data)
        self._idx = 0
        return self

    def __next__(self):
        if self._epoch >= self._num_epochs:
            raise StopIteration

        x = self._data[self._idx: self._idx + self._batch_size]
        self._idx += self._batch_size

        if self._idx <= self._data_len:
            assert x.shape[0] == self._batch_size
            return x

        self._epoch += 1
        if self._epoch >= self._num_epochs:
            if not self._allow_smaller or x.shape[0] == 0:
                raise StopIteration
            return x

        # Reshuffle the data again
        if self._shuffle:
            np.random.shuffle(self._data)
        self._idx = self._batch_size - x.shape[0]

        return np.concatenate((x, self._data[:self._idx]), axis=0)

    def next(self):
        return self.__next__()

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def epoch(self):
        return self._epoch

    @property
    def batch_size(self):
        return self._batch_size


class FileBatchIterator:
    """CSV file batch retrieval tool"""
    def __init__(self, filenames,
                 delimiter=None,
                 skip_rows=0,
                 batch_size=100,
                 allow_smaller_batch=False,
                 num_epochs=None,
                 shuffle=False,
                 shuffle_factor=5):
        """

        :param filenames:
        :param delimiter:
        :param skip_rows:
        :param batch_size:
        :param allow_smaller_batch:
        :param num_epochs:
        :param shuffle:
        :param shuffle_factor:
        """
        self._files = [filenames] if isinstance(filenames, str) else filenames
        self._num_epochs = num_epochs if num_epochs is not None else MAX_EPOCHS
        self._batch_size = batch_size
        self._delimiter = delimiter
        self._skip_rows = skip_rows
        self._load_size = self._batch_size if not shuffle else self._batch_size * shuffle_factor
        self._shuffle = shuffle
        self._allow_smaller = allow_smaller_batch
        self.__iter__()

    def __iter__(self):
        if self._shuffle:
            rnd.shuffle(self._files)
        self._current = 0
        self._prepare_file()
        self._epoch = 0
        self._idx = 0
        self._prefetched = None
        return self

    def _next_file(self):
        self._current += 1
        if self._current >= len(self._files):
            self._epoch += 1
            if self._epoch >= self._num_epochs:
                return False
            if self._shuffle:
                rnd.shuffle(self._files)
            self._current = 0

        if len(self._files) == 1:
            self._fh.seek(0)
        else:
            self._fh.close()
            self._prepare_file()
        return True

    def _prepare_file(self):
        self._fh = open(self._files[self._current], "rt")
        for i in range(self._skip_rows):
            self._fh.readline()

    def _get_lines(self, max_lines=None):
        batch = []

        cnt = self._load_size if max_lines is None else max_lines
        while cnt > 0:
            line = self._fh.readline()
            if not line:
                if not self._next_file():
                    break
                else:
                    cnt += 1
            else:
                batch.append(np.fromstring(line, dtype=np.float32, sep=self._delimiter or ' '))
            cnt -= 1

        return np.array(batch)

    def _finish(self):
        if self._fh is not None:
            self._fh.close()
        raise StopIteration

    def __next__(self):
        if self._epoch >= self._num_epochs:
            x = None
            self._finish()
        elif not self._shuffle:
            x = self._get_lines()
        elif self._prefetched is None:
            self._prefetched = self._get_lines()
            np.random.shuffle(self._prefetched)
            self._idx = self._batch_size
            x = self._prefetched[0:self._batch_size]
        else:
            x = self._prefetched[self._idx:self._idx + self._batch_size]
            self._idx += self._batch_size
            if self._idx == self._load_size:
                self._prefetched = None

        if x.shape[0] == 0 or (not self._allow_smaller and x.shape[0] < self._batch_size):
            self._finish()
        else:
            return x

    def next(self):
        return self.__next__()

    @property
    def epoch(self):
        return self._epoch

    @property
    def batch_size(self):
        return self._batch_size


def dt_delimiter_from_ext(f_ext):
    if f_ext == ".txt":
        return ' '
    elif f_ext == ".csv":
        return ','
    elif f_ext == ".tsv":
        return ","
    else:
        return None


def dt_prepare_iterator(filename,
                        delimiter=None,
                        skip_rows=0,
                        batch_size=100,
                        shuffle=False,
                        num_epochs=None,
                        allow_smaller_batch=False):
    """
    Prepare a data batch iterator which can be used in various places...
    :param filename: The list of files to be used, or the np.array.
    :param in_memory: Whether to try to load everything into the memory.
    :param delimiter: The file delimiter if that is the case.
    :param skip_rows: How many lines to skip, in the case of file format.
    :param batch_size: The batch size to be used.
    :param shuffle: Whether to shuffle - both filenames and the data.
    :param num_epochs: Number of epochs to iterate over the files.
    :param allow_smaller_batch: Whether the last batch can be smaller.
    :return: An iterator.
    """
    if filename is None:
        return None
    _, f_ext = os.path.splitext(filename)

    if f_ext == ".npy":
        return ArrayBatchIterator(np.load(filename),
                              batch_size=batch_size,
                              allow_smaller_batch=allow_smaller_batch,
                              num_epochs=num_epochs,
                              shuffle=shuffle)
    elif f_ext == ".gz":
        img_data, num, w, h = packed_images_reader(filename)
        return ArrayBatchIterator(np.reshape(img_data, newshape=[num, w * h]),
                              batch_size=batch_size,
                              allow_smaller_batch=allow_smaller_batch,
                              num_epochs=num_epochs,
                              shuffle=shuffle)
    elif os.path.getsize(filename) < DEF_MAX_MEMORY_SIZE:
        if delimiter is None:
            delimiter = dt_delimiter_from_ext(f_ext)
        return ArrayBatchIterator(np.loadtxt(filename, delimiter=delimiter, skiprows=skip_rows),
                              batch_size=batch_size,
                              allow_smaller_batch=allow_smaller_batch,
                              num_epochs=num_epochs,
                              shuffle=shuffle)
    else:
        return FileBatchIterator(filename,
                             delimiter=delimiter,
                             skip_rows=skip_rows,
                             batch_size=batch_size,
                             allow_smaller_batch=allow_smaller_batch,
                             num_epochs=num_epochs,
                             shuffle=shuffle)


if __name__ == '__main__':
    files = FileBatchIterator(['file1.txt', 'file2.txt', 'file3.txt'],
                          batch_size=7,
                          allow_smaller_batch=True,
                          num_epochs=5,
                          shuffle=True)
    with open("outfile.txt", "wt") as outf:
        for row in files:
            np.savetxt(outf, row, fmt="%.1f")
            outf.write("----\n")
