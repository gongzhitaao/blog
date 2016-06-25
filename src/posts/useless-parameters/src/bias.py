import numpy as np
import pandas as pd

from keras.models import model_from_json

# import matplotlib
# matplotlib.use('Qt5Agg')
# from matplotlib import pyplot as plt
# import matplotlib.gridspec as gridspec
# matplotlib.style.use('ggplot')


from mnist import make_dataset


DATA_PATH = '/home/zzg0009/data/mnist/'
db = make_dataset(DATA_PATH)
W, H = db.image_size
test = db.test

model = model_from_json(open('model/fc100-100-10.json').read())
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adagrad')

model.load_weights('result/fc100-100-10-weights-99.h5')


def shuffle_bias(n):
    """Shuffle bias of a layer."""

    res = model.evaluate(test.data, test.target)
    print('loss: {0:.4f}, acc: {1:.4f}'.format(res[0], res[1]))

    layer = model.layers[n]
    acc = []
    for i in np.arange(100):
        print('Try {0:02d}...'.format(i))
        w, b = layer.get_weights()
        np.random.shuffle(b)
        layer.set_weights([w, b])
        res = model.evaluate(test.data, test.target, verbose=0)
        print('loss: {0:.4f}, acc: {1:.4f}'.format(res[0], res[1]))
        acc.append(res[1])

    print('Acc mean: {0}, std: {1}'.format(np.mean(acc), np.std(acc)))


def shuffle_all_bias():
    """Shuffle all bias."""
    res = model.evaluate(test.data, test.target)
    print('loss: {0:.4f}, acc: {1:.4f}'.format(res[0], res[1]))

    layers = [model.layers[0], model.layers[2], model.layers[4]]

    bias = []
    for layer in layers:
        w, b = layer.get_weights()
        bias = np.append(bias, b)

    acc = []
    for i in np.arange(100):
        print('Try {0:02d}...'.format(i))
        np.random.shuffle(bias)

        start = 0
        for layer in layers:
            w, b = layer.get_weights()
            b = bias[start : (start+b.shape[0])]
            layer.set_weights([w, b])
            start += b.shape[0]

        res = model.evaluate(test.data, test.target, verbose=0)
        print('loss: {0:.4f}, acc: {1:.4f}'.format(res[0], res[1]))
        acc.append(res[1])

    print('Acc mean: {0}, std: {1}'.format(np.mean(acc), np.std(acc)))


def reset_bias(n):
    """Reset bias to random values in different range."""

    res = model.evaluate(test.data, test.target)
    print('loss: {0:.4f}, acc: {1:.4f}'.format(res[0], res[1]))

    layer = model.layers[n]

    acc_mean = []
    acc_std = []
    for m in np.arange(100):
        acc = []
        for i in np.arange(100):
            print('Try {0:02d}...'.format(i), end='')
            w, b = layer.get_weights()
            b = np.random.random(b.shape) * m
            layer.set_weights([w, b])
            res = model.evaluate(test.data, test.target, verbose=0)
            print('loss: {0:.4f}, acc: {1:.4f}'
                  .format(res[0], res[1]), end='\r')
            acc.append(res[1])

        acc_mean.append(np.mean(acc))
        acc_std.append(np.std(acc))

        print('Max {2:02d} acc mean: {0:.4f}, std: {1:.6f}'
              .format(acc_mean[-1], acc_std[-1], m))

    df = pd.DataFrame({'range': np.arange(100),
                       'accuracy_mean': acc_mean,
                       'accuracy_std': acc_std})
    df.to_csv('reset_bias_{0}.csv'.format(n), index=False)


def reset_all_bias():
    """Reset all bias to random values in different range."""

    res = model.evaluate(test.data, test.target)
    print('loss: {0:.4f}, acc: {1:.4f}'.format(res[0], res[1]))

    layers = [model.layers[0], model.layers[2], model.layers[4]]

    acc_mean = []
    acc_std = []
    for m in np.arange(100):
        acc = []
        for i in np.arange(100):
            print('Try {0:02d}...'.format(i), end='')

            for layer in layers:
                w, b = layer.get_weights()
                b = np.random.random(b.shape) * m
                layer.set_weights([w, b])

            res = model.evaluate(test.data, test.target, verbose=0)
            print('loss: {0:.4f}, acc: {1:.4f}'
                  .format(res[0], res[1]), end='\r')
            acc.append(res[1])

        acc_mean.append(np.mean(acc))
        acc_std.append(np.std(acc))

        print('Max {2:02d} acc mean: {0:.4f}, std: {1:.6f}'
              .format(acc_mean[-1], acc_std[-1], m))

    df = pd.DataFrame({'range': np.arange(100),
                       'accuracy_mean': acc_mean,
                       'accuracy_std': acc_std})
    df.to_csv('reset_bias_all.csv', index=False)


if __name__ == '__main__':
    # shuffle_bias(0)
    # shuffle_all_bias()
    # reset_bias(4)
    # reset_all_bias()
