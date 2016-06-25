"""Interface for MNIST."""

import gzip
import os

import numpy as np

def _read32(bytestream, dtype=np.dtype(np.uint32).newbyteorder('>')):
    return np.frombuffer(bytestream.read(4), dtype=dtype)[0]


def extract_images(filename):
    """Extract images."""
    print('Extracting', filename)
    with gzip.open(filename, 'rb') as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows * cols)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert from one-hot represention."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract labels."""
    print('Extracting', filename)
    with gzip.open(filename, 'rb') as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
        return dense_to_one_hot(labels)
    return labels


class DataSet(object):
    """Interface for MNIST"""

    BATCH_SIZE = 10

    def __init__(self, images, labels):
        self._data = images
        self._target = labels

        self._num_samples = self._data.shape[0]
        self._completed_epochs = 0
        self._sample_index = 0

    @property
    def data(self):
        """Get the image data."""
        return self._data

    @property
    def target(self):
        """Get label data."""
        return self._target

    @property
    def num_samples(self):
        """Total number of samples."""
        return self._num_samples

    def next_batch(self, batch_size=None):
        """Next batch of data."""
        if batch_size is None:
            batch_size = self.BATCH_SIZE

        end = self._sample_index + batch_size
        if end > self._num_samples:
            self._completed_epochs += 1
            self._sample_index = 0

            perm = np.arange(self._num_samples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._target = self._target[perm]

            return None, None

        start = self._sample_index
        end = start + batch_size

        self._sample_index = end

        return self._data[start:end], self._target[start:end]


def make_dataset(datapath, validation=None, one_hot=True):
    class DataSets(object):
        pass
    ret = DataSets()

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    train_images = extract_images(os.path.join(datapath, TRAIN_IMAGES))
    train_images = train_images / 255.
    train_labels = extract_labels(os.path.join(datapath, TRAIN_LABELS),
                                  one_hot=one_hot)

    test_images = extract_images(os.path.join(datapath, TEST_IMAGES))
    test_images = test_images / 255.
    test_labels = extract_labels(os.path.join(datapath, TEST_LABELS),
                                 one_hot=one_hot)

    if validation is not None:
        VALIDATION_SIZE = int(train_images.shape[0] * validation)
        validation_images = train_images[:VALIDATION_SIZE]
        validation_labels = train_labels[:VALIDATION_SIZE]
        train_images = train_images[VALIDATION_SIZE:]
        train_labels = train_labels[VALIDATION_SIZE:]
        ret.validation = DataSet(validation_images, validation_labels)

    ret.train = DataSet(train_images, train_labels)
    ret.test = DataSet(test_images, test_labels)
    ret.n_input = train_images.shape[1]
    ret.n_output = train_labels.shape[1]
    ret.image_size = (28, 28)

    return ret
