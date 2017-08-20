#
# General Data handling utils, like file and np array iterators.
#
# Copyright (C) 2017, IDEAConsult Ltd.
#
# Author: Ivan (Jonan) Georgiev

import numpy as np
import random as rnd
from .nn_utils import packed_images_reader

DT_FORMAT_MEM = "array"
DT_FORMAT_FILE = "file"
DT_FORMAT_IMAGE = "images"
MAX_EPOCHS = 100000


class ArrayBatchIter:
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


class FileBatchIter:
    """CSV file batch retrieval tool"""
    def __init__(self, filenames,
                 delimiter=None,
                 batch_size=100,
                 allow_smaller_batch=False,
                 num_epochs=None,
                 shuffle=False,
                 shuffle_factor=5):
        """

        :param filename:
        :param delimiter:
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
        self._load_size = self._batch_size if not shuffle else self._batch_size * shuffle_factor
        self._shuffle = shuffle
        self._allow_smaller = allow_smaller_batch
        self.__iter__()

    def __iter__(self):
        if self._shuffle:
            rnd.shuffle(self._files)
        self._current = 0
        self._fh = open(self._files[0], "rt")
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
            self._fh = open(self._files[self._current], "rt")
        return True

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


def dt_prepare_iterator(datas,
                        data_format=DT_FORMAT_FILE,
                        delimiter=None,
                        batch_size=100,
                        shuffle=False,
                        num_epochs=None,
                        allow_smaller_batch=False):
    """
    Prepare a data batch iterator which can be used in various places...
    :param datas: The list of files to be used, or the np.array.
    :param data_format: The given format - one of DT_* constants.
    :param delimiter: The file delimiter if that is the case.
    :param batch_size: The batch size to be used.
    :param shuffle: Whether to shuffle - both filenames and the data.
    :param num_epochs: Number of epochs to iterate over the files.
    :param allow_smaller_batch: Whether the last batch can be smaller.
    :return: An iterator.
    """
    if datas is None:
        return None
    elif data_format == DT_FORMAT_IMAGE:
        img_data, num, w, h = packed_images_reader(datas[0])
        return ArrayBatchIter(np.reshape(img_data, newshape=[num, w * h]),
                              batch_size=batch_size,
                              allow_smaller_batch=allow_smaller_batch,
                              num_epochs=num_epochs,
                              shuffle=shuffle)
    elif data_format == DT_FORMAT_MEM:
        return ArrayBatchIter(np.loadtxt(datas[0], delimiter=delimiter),
                              batch_size=batch_size,
                              allow_smaller_batch=allow_smaller_batch,
                              num_epochs=num_epochs,
                              shuffle=shuffle)
    elif data_format == DT_FORMAT_FILE:
        return FileBatchIter(datas,
                             delimiter=delimiter,
                             batch_size=batch_size,
                             allow_smaller_batch=allow_smaller_batch,
                             num_epochs=num_epochs,
                             shuffle=shuffle)
    else:
        return None


if __name__ == '__main__':
    files = FileBatchIter(['file1.txt', 'file2.txt', 'file3.txt'],
                          batch_size=7,
                          allow_smaller_batch=True,
                          num_epochs=5,
                          shuffle=True)
    with open("outfile.txt", "wt") as outf:
        for row in files:
            np.savetxt(outf, row, fmt="%.1f")
            outf.write("----\n")
