from itertools import tee

import numpy as np
import pandas as pd

from keras.models import model_from_json

from mnist import make_dataset


DATA_PATH = '/home/zzg0009/data/mnist/'
db = make_dataset(DATA_PATH)
W, H = db.image_size
test = db.test

model = model_from_json(open('model/fc100-100-10.json').read())
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adagrad')

model.load_weights('result/fc100-100-10-weights-99.h5')


weights = pd.DataFrame(np.load('weights.npy'))
size = [28 * 28, 100, 100, 10]

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def shuffle_all_weight(n):
    """Shuffle all weight of a layer."""

    res = model.evaluate(test.data, test.target)
    print('loss: {0:.4f}, acc: {1:.4f}'.format(res[0], res[1]))

    layer = model.layers[n]
    acc = []
    for i in np.arange(100):
        print('Try {0:02d}...'.format(i), end='')
        w, b = layer.get_weights()
        np.random.shuffle(w)
        layer.set_weights([w, b])
        res = model.evaluate(test.data, test.target, verbose=0)
        print('loss: {0:.4f}, acc: {1:.4f}'.format(res[0], res[1]))
        acc.append(res[1])
    print('Acc mean: {0}, std: {1}'.format(np.mean(acc), np.std(acc)))


def shuffle_bad_learner(n):
    """Shuffle weights that does not learn much."""

    a, b = weights.iloc[0], weights.iloc[-1]
    diff = np.absolute((b - a) / a * 100)

    start = 0
    for i, (l0, l1) in enumerate(pairwise(size)):
        if i < n:
            start += l0 * l1 + l1
        else:
            break

    acc_mean = []
    acc_std = []
    count = []
    layer = model.layers[n * 2]
    total = l0 * l1

    q0, q1 = np.percentile(diff, [25, 50])
    d = q1 - q0
    # q0 -= 1.5 * d
    # q1 += 1.5 * d

    bounds = np.arange(0, q1, 1.0)
    N = 100

    for thres in bounds:
        indices = diff <= thres
        ind, = np.where(indices[start : (start+l0*l1)])
        count.append(ind.shape[0] / total)

        acc = []
        for i in np.arange(N):
            print('Thres {0:.1f} count {2:.4f}% try {1:02d}/{3}...'
                  .format(thres, i, count[-1] * 100, N-1), end='')
            w, b = layer.get_weights()

            shape = w.shape
            w = w.flatten()
            ww = w[ind]
            np.random.shuffle(ww)
            w[ind] = ww
            w = w.reshape(shape)

            layer.set_weights([w, b])
            res = model.evaluate(test.data, test.target, verbose=0)
            print('loss: {0:.4f}, acc: {1:.4f}'
                  .format(res[0], res[1], thres), end='\r')
            acc.append(res[1])

        acc_mean.append(np.mean(acc))
        acc_std.append(np.std(acc))

        print()

    df = pd.DataFrame({'range': bounds,
                       'accuracy_mean': acc_mean,
                       'accuracy_std': acc_std,
                       'count': count})
    df.to_csv('shuffle_bad_learner_{0}.csv'.format(n), index=False)


def shuffle_all_bad_learner():
    """Shuffle all weights that does not learn much."""

    a, b = weights.iloc[0], weights.iloc[-1]
    diff = np.absolute((b - a) / a * 100)

    acc_mean = []
    acc_std = []
    count = []
    q = np.percentile(diff, 50)
    N = 100
    bounds = np.arange(0, q, 1.0)

    layers = [model.layers[0], model.layers[2], model.layers[4]]
    for thres in bounds:
        indices = diff <= thres
        ind = []
        start = 0
        bad = []

        for l0, l1 in pairwise(size):
            ind.append(np.where(indices[start : (start+l0*l1)])[0])
            start += l0 * l1 + l1
        ind = np.array(ind)

        for i, layer in enumerate(layers):
            w, b = layer.get_weights()
            bad = np.append(bad, w.flatten()[ind[i]])

        count.append(bad.shape[0] / indices.shape[0])

        acc = []
        for i in np.arange(N):
            print('Thres {0:.1f} count {2:.4f}% try {1:02d}/{3}...'
                  .format(thres, i, count[-1] * 100, N-1), end='')

            np.random.shuffle(bad)

            start = 0
            for j, layer in enumerate(layers):
                w, b = layer.get_weights()
                shape = w.shape
                w = w.flatten()
                w[ind[j]] = bad[start : (start+ind[j].shape[0])]
                w = w.reshape(shape)
                layer.set_weights([w, b])
                start += ind[j].shape[0]

            res = model.evaluate(test.data, test.target, verbose=0)
            print('loss: {0:.4f}, acc: {1:.4f}'
                  .format(res[0], res[1], thres), end='\r')
            acc.append(res[1])

        acc_mean.append(np.mean(acc))
        acc_std.append(np.std(acc))

        print()

    df = pd.DataFrame({'range': bounds,
                       'accuracy_mean': acc_mean,
                       'accuracy_std': acc_std,
                       'count': count})
    df.to_csv('shuffle_bad_learner_all.csv', index=False)


def reset_bad_learner(n, U):
    a, b = weights.iloc[0], weights.iloc[-1]
    diff = np.absolute((b - a) / a * 100)

    start = 0
    for i, (l0, l1) in enumerate(pairwise(size)):
        if i < n:
            start += l0 * l1 + l1
        else:
            break

    acc_mean = []
    acc_std = []
    count = []
    layer = model.layers[n * 2]
    total = l0 * l1

    q = np.percentile(diff, 50)

    bounds = np.arange(0, q, 1.0)
    N = 100

    for thres in bounds:
        indices = diff <= thres
        ind, = np.where(indices[start : (start+l0*l1)])
        count.append(ind.shape[0] / total)

        acc = []
        for i in np.arange(N):
            print('Thres {0:.1f} count {2:.4f}% try {1:02d}/{3}...'
                  .format(thres, i, count[-1] * 100, N-1), end='')
            w, b = layer.get_weights()

            shape = w.shape
            w = w.flatten()
            w[ind] = np.random.random((ind.shape[0],)) * U
            w = w.reshape(shape)

            layer.set_weights([w, b])
            res = model.evaluate(test.data, test.target, verbose=0)
            print('loss: {0:.4f}, acc: {1:.4f}'
                  .format(res[0], res[1], thres), end='\r')
            acc.append(res[1])

        acc_mean.append(np.mean(acc))
        acc_std.append(np.std(acc))

        print()

    df = pd.DataFrame({'range': bounds,
                       'accuracy_mean': acc_mean,
                       'accuracy_std': acc_std,
                       'count': count})
    df.to_csv('reset_bad_learner_{0}_{1:.1f}.csv'.format(n, U),
              index=False)


def reset_all_bad_learner(U):
    """Shuffle all weights that does not learn much."""

    a, b = weights.iloc[0], weights.iloc[-1]
    diff = np.absolute((b - a) / a * 100)

    acc_mean = []
    acc_std = []
    count = []
    q = np.percentile(diff, 50)
    N = 100
    bounds = np.arange(0, q, 1.0)

    layers = [model.layers[0], model.layers[2], model.layers[4]]
    for thres in bounds:
        indices = diff <= thres
        ind = []
        start = 0
        bad = []

        total = 0
        for l0, l1 in pairwise(size):
            ind.append(np.where(indices[start : (start+l0*l1)])[0])
            start += l0 * l1 + l1
            total += ind[-1].shape[0]
        ind = np.array(ind)

        count.append(total / indices.shape[0])

        acc = []
        for i in np.arange(N):
            print('Thres {0:.1f} count {2:.4f}% try {1:02d}/{3}...'
                  .format(thres, i, count[-1] * 100, N-1), end='')

            bad = np.random.random((total,)) * U

            start = 0
            for j, layer in enumerate(layers):
                w, b = layer.get_weights()
                shape = w.shape
                w = w.flatten()
                w[ind[j]] = bad[start : (start+ind[j].shape[0])]
                w = w.reshape(shape)
                layer.set_weights([w, b])
                start += ind[j].shape[0]

            res = model.evaluate(test.data, test.target, verbose=0)
            print('loss: {0:.4f}, acc: {1:.4f}'
                  .format(res[0], res[1], thres), end='\r')
            acc.append(res[1])

        acc_mean.append(np.mean(acc))
        acc_std.append(np.std(acc))

        print()

    df = pd.DataFrame({'range': bounds,
                       'accuracy_mean': acc_mean,
                       'accuracy_std': acc_std,
                       'count': count})
    df.to_csv('reset_bad_learner_all_{0:.1f}.csv'.format(U),
              index=False)


def final_result():
    """Compress the model as much as possible.

    1. Reset all biases to zero
    2. Reset all weights with change scale less 20% to zero.
    """

    res = model.evaluate(test.data, test.target)
    print('loss: {0:.4f}, acc: {1:.4f}' .format(res[0], res[1]))

    a, b = weights.iloc[0], weights.iloc[-1]
    diff = np.absolute((b - a) / a * 100)

    acc = [res[1]]
    nparam = [weights.iloc[0].size]
    bounds = np.arange(30)

    for thres in bounds[1:]:
        indices = diff <= thres
        count = 0

        layers = [model.layers[0], model.layers[2], model.layers[4]]
        start = 0
        for i, (l0, l1) in enumerate(pairwise(size)):
            w, b = layers[i].get_weights()

            ind = np.where(indices[start : (start+l0*l1)])[0]
            shape = w.shape
            w = w.flatten()
            w[ind] = 0.
            w = w.reshape(shape)
            count += ind.size

            b.fill(0.)
            count += b.size

            layers[i].set_weights([w, b])

            start += l0 * l1 + l1

        res = model.evaluate(test.data, test.target, verbose=0)
        print('loss: {0:.4f}, acc: {1:.4f}' .format(res[0], res[1]))
        print('# param: {0}, # param: {1} ({2:.3f}))'.format(
            weights.iloc[0].size,
            weights.iloc[0].size - count,
            1 - count / weights.iloc[0].size
        ))

        acc.append(res[1])
        nparam.append(weights.iloc[0].size - count)

    df = pd.DataFrame({'accuracy': acc,
                       'nparam': nparam,
                       'bounds': bounds})
    df.to_csv('final_result.csv', index=False)

if __name__ == '__main__':
    # shuffle_all_weight(0)
    # shuffle_bad_learner(0)
    # shuffle_bad_learner(1)
    # shuffle_bad_learner(2)
    # shuffle_all_bad_learner()
    # reset_bad_learner(2, 1)
    # reset_all_bad_learner(1)
    final_result()
