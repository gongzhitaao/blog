import os
import sys
import logging

import numpy as np
import pandas as pd

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint

from mnist import make_dataset


fmt = logging.Formatter("%(asctime)s:%(message)s")
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(ch)


DATA_PATH = '/home/zzg0009/data/mnist/'
db = make_dataset(DATA_PATH)
W, H = db.image_size

def prepare_model(topology=None, weight=None, force=False):

    # build model

    if topology is not None and os.path.isfile(topology):
        model = model_from_json(open(topology).read())
    else:
        model = Sequential()
        model.add(Dense(input_dim=W * H, output_dim=100, bias=False))
        model.add(Activation('sigmoid'))
        model.add(Dense(input_dim=100, output_dim=100, bias=False))
        model.add(Activation('sigmoid'))
        model.add(Dense(input_dim=100, output_dim=10, bias=False))
        model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer='adagrad')

    with open(topology, 'w') as w:
        w.write(model.to_json())

    # train it on mnist

    model.save_weights('{0}-0'.format(weight), overwrite=True)

    if not force and \
       weight is not None and \
       os.path.isfile(weight):
        model.load_weights(weight)
    else:
        train = db.train
        checkpointer = ModelCheckpoint(
            filepath='result/fc100-100-10-weights-{epoch:02d}.h5',
            verbose=1)
        model.fit(train.data, train.target, nb_epoch=100,
                  validation_split=0.1, callbacks=[checkpointer])
        model.save_weights('{0}-1'.format(weight), overwrite=True)

    return model


# merge all weights together
def merge_weights():
    model = model_from_json(open('model/fc100-100-10.json').read())
    weights = []
    for fn in sorted(os.listdir('result')):
        model.load_weights(os.path.join('result', fn))
        row = []
        for layer in model.layers:
            for p in layer.get_weights():
                row = np.append(row, p)
        weights.append(row)
    weights = np.array(weights)
    np.save('weights', weights)


if __name__ == '__main__':
    # prepare_model(topology='model/fc100-100-10.json',
    #           weight='model/fc100-100-10.h5', force=True)

    merge_weights()
