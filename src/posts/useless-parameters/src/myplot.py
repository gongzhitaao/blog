import math
import os
from itertools import tee

import numpy as np
import pandas as pd

from keras.models import model_from_json

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.style.use('ggplot')
import matplotlib.animation as animation

weights = pd.DataFrame(np.load('weights.npy'))
size = [28 * 28, 100, 100, 10]


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


d = []
start = 0
for a, b in pairwise(size):
    l0 = a * b
    l1 = b
    d.append((start, l0, l1))
    start += l0 + l1


def distribution_per_layer():
    dpi = 200
    fig = plt.figure(figsize=(8, 4), dpi=dpi)

    gs = gridspec.GridSpec(2, 2)
    ax = []
    for i in np.arange(2):
        for j in np.arange(2):
            ax.append(fig.add_subplot(gs[i, j]))

    def _plot(param, ax0, ax1):
        for i, (a, b, c) in enumerate(d):
            w = param.ix[a:(a+b)]
            w.plot(kind='kde', ax=ax0, label='Layer {0}'.format(i))
            b = param.ix[(a+b):(a+b+c)]
            b.plot(kind='kde', ax=ax1, label='Layer {0}'.format(i))

    _plot(weights.iloc[0], ax[0], ax[1])
    ax[0].set_xlim([-1.0, 1.0])
    ax[1].set_xlim([-0.5, 1.0])

    _plot(weights.iloc[-1], ax[2], ax[3])
    ax[2].set_xlim([-1.5, 1.5])
    ax[3].set_xlim([-0.5, 1.5])


    ax[0].set_title('Weight Density')
    ax[0].set_ylabel('Epoch 0', rotation=0, size='large')
    ax[0].yaxis.set_label_coords(-0.3, 0.5)
    ax[1].set_ylabel('')
    ax[1].set_title('Bias Density')
    ax[2].set_ylabel('Epoch 100', rotation=0, size='large')
    ax[2].yaxis.set_label_coords(-0.3, 0.5)
    ax[3].set_ylabel('')
    ax[3].legend()

    plt.tight_layout()
    plt.savefig('fc100-100-10-weight-per-layer.png', format='png',
                dpi=dpi)

# distribution_per_layer()

def how_much_learned_kde():
    dpi = 200
    fig = plt.figure(figsize=(8, 4), dpi=dpi)

    gs = gridspec.GridSpec(2, 2)
    ax = []
    for i in np.arange(2):
        for j in np.arange(2):
            ax.append(fig.add_subplot(gs[i, j]))

    def _plot_kde(param, ax0, ax1):
        for i, (a, b, c) in enumerate(d):
            w = param.ix[a:(a+b)]
            w.plot(kind='kde', ax=ax0, label='Layer {0}'.format(i))
            b = param.ix[(a+b):(a+b+c)]
            b.plot(kind='kde', ax=ax1, label='Layer {0}'.format(i))

    diff = weights.iloc[-1] - weights.iloc[0]
    _plot_kde(diff, ax[0], ax[1])
    ax[0].set_xlim([-1.0, 1.0])
    ax[1].set_xlim([-0.5, 0.5])

    diff = diff / weights.iloc[0]
    _plot_kde(diff, ax[2], ax[3])
    ax[2].set_xlim([-1e2, 1e2])
    yfmt = matplotlib.ticker.ScalarFormatter(useOffset=True)
    yfmt.set_powerlimits((-2, 1))
    ax[2].yaxis.set_major_formatter(yfmt)
    ax[3].set_xlim([-1e2, 1e2])

    ax[0].set_title('Weight Diff Density')
    ax[0].set_ylabel('Absolute', rotation=0, size='large')
    ax[0].yaxis.set_label_coords(-0.3, 0.5)
    ax[1].set_ylabel('')
    ax[1].set_title('Bias Diff Density')
    ax[2].set_ylabel('Relative', rotation=0, size='large')
    ax[2].yaxis.set_label_coords(-0.3, 0.5)
    ax[3].set_ylabel('')
    ax[3].legend()

    plt.tight_layout()
    plt.savefig('fc100-100-10-weight-diff-kde.png', format='png',
                dpi=dpi)


def how_much_learned_box():
    dpi = 200
    fig = plt.figure(figsize=(8, 4), dpi=dpi)
    gs = gridspec.GridSpec(3, 2)
    ax = []
    for i in np.arange(2):
        for j in np.arange(3):
            ax.append(fig.add_subplot(gs[j, i]))

    def _plot_box(param, ax0, ax1):

        for i, (a, b, c) in enumerate(d):
            w = param.ix[a:(a+b)]

            w.plot(kind='box', ax=ax0[i], label='', vert=False,
                   sym='b.')
            b = param.ix[(a+b):(a+b+c)]
            b.plot(kind='box', ax=ax1[i], label='', vert=False,
                   sym='b.')
            ax0[i].set_ylabel('Layer {0}'.format(i), rotation=0)
            ax0[i].yaxis.set_label_coords(-0.15, 0.5)

            xfmt = matplotlib.ticker.ScalarFormatter(useOffset=True)
            xfmt.set_powerlimits((-2, 2))
            ax0[i].xaxis.set_major_formatter(xfmt)

            xfmt = matplotlib.ticker.ScalarFormatter(useOffset=True)
            xfmt.set_powerlimits((-1, 1))
            ax1[i].xaxis.set_major_formatter(xfmt)

        ax0[0].set_title('Weight', size='large')
        ax1[0].set_title('Bias', size='large')

    diff = weights.iloc[-1] - weights.iloc[0]
    diff = diff / weights.iloc[0] * 100
    _plot_box(diff, ax[:3], ax[3:])

    plt.tight_layout()
    plt.savefig('fc100-100-10-weight-diff-box.png', format='png',
                dpi=dpi)


def how_much_outlier_perc1():
    diff = weights.iloc[-1] - weights.iloc[0]
    diff /= weights.iloc[0] * 100

    with open('tmp.txt', 'w') as wf:
        for i, (a, b, c) in enumerate(d):
            w = diff.ix[a:(a+b)].as_matrix().flatten()
            b = diff.ix[(a+b):(a+b+c)].as_matrix().flatten()

            c0, n0 = np.sum(w > 1), w.shape[0]
            c1, n1 = np.sum(b > 1), b.shape[0]
            fmt = '{0}/{1} ({2:.4f}%)'
            wf.write('| {0} | {1} |\n'.format(
                fmt.format(c0, n0, c0 / n0 * 100.),
                fmt.format(c1, n1, c1 / n1 * 100.)))

# how_much_learned_outlier()


def how_much_outlier_quantile():
    diff = weights.iloc[-1] - weights.iloc[0]
    diff /= weights.iloc[0] * 100

    with open('tmp1.txt', 'w') as wf:
        for i, (a, b, c) in enumerate(d):
            w = diff.ix[a:(a+b)].as_matrix().flatten()
            b = diff.ix[(a+b):(a+b+c)].as_matrix().flatten()

            q = np.percentile(w, [25, 75])
            p = [q[0] - 1.5*(q[1]-q[0]), q[1] + 1.5*(q[1]-q[0])]
            c0 = np.sum(np.logical_or(w < p[0], w > p[1]))
            n0 = w.shape[0]

            q = np.percentile(b, [25, 75])
            p = [q[0] - 1.5*(q[1]-q[0]), q[1] + 1.5*(q[1]-q[0])]
            c1 = np.sum(np.logical_or(b < p[0], b > p[1]))
            n1 = b.shape[0]
            fmt = '{0}/{1} ({2:.4f}%)'
            wf.write('| {0} | {1} |\n'.format(
                fmt.format(c0, n0, c0 / n0 * 100.),
                fmt.format(c1, n1, c1 / n1 * 100.)))


def how_much_learned_kde_absolute():
    dpi = 200
    fig = plt.figure(figsize=(8, 4), dpi=dpi)

    gs = gridspec.GridSpec(2, 2)
    ax = []
    for i in np.arange(2):
        for j in np.arange(2):
            ax.append(fig.add_subplot(gs[i, j]))

    def _plot_kde(param, ax0, ax1):
        for i, (a, b, c) in enumerate(d):
            w = param.ix[a:(a+b)]
            w.plot(kind='kde', ax=ax0, label='Layer {0}'.format(i))
            b = param.ix[(a+b):(a+b+c)]
            b.plot(kind='kde', ax=ax1, label='Layer {0}'.format(i))

    diff = np.absolute(weights.iloc[-1] - weights.iloc[0])
    _plot_kde(diff, ax[0], ax[1])
    ax[0].set_xlim([-1.0, 1.0])
    ax[1].set_xlim([-0.5, 0.5])

    diff = np.absolute(diff / weights.iloc[0])
    _plot_kde(diff, ax[2], ax[3])
    ax[2].set_xlim([-1e2, 1e2])
    yfmt = matplotlib.ticker.ScalarFormatter(useOffset=True)
    yfmt.set_powerlimits((-2, 1))
    ax[2].yaxis.set_major_formatter(yfmt)
    ax[3].set_xlim([-1e2, 1e2])

    ax[0].set_title('Weight Diff Density')
    ax[0].set_ylabel('Absolute', rotation=0, size='large')
    ax[0].yaxis.set_label_coords(-0.3, 0.5)
    ax[1].set_ylabel('')
    ax[1].set_title('Bias Diff Density')
    ax[2].set_ylabel('Relative', rotation=0, size='large')
    ax[2].yaxis.set_label_coords(-0.3, 0.5)
    ax[3].set_ylabel('')
    ax[3].legend()

    plt.tight_layout()
    plt.savefig('fc100-100-10-weight-diff-kde-absolute.png', format='png',
                dpi=dpi)


def reset_bias():
    dpi = 200
    fig = plt.figure(figsize=(16, 4), dpi=dpi)
    gs = gridspec.GridSpec(1, 4)

    color = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

    N = 21

    def _plot(df, ax, label, color):
        df.plot(x='range', y='accuracy_mean', ax=ax, label=label,
                c=color)
        x = df['range']
        y = df['accuracy_mean']
        e = df['accuracy_std']
        plt.fill_between(x, y-e, y+e, alpha=0.1, edgecolor='#1B2ACC',
                         facecolor='#089FFF', antialiased=True)
        ax.set_xlabel('')

    for n in np.arange(3):
        if n > 0:
            ax0 = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[n], sharey=ax0)
        else:
            ax = fig.add_subplot(gs[n])
        df = pd.read_csv('reset_bias_{0}.csv'.format(n * 2))
        df = df.head(N)
        _plot(df, ax, 'Layer {0}'.format(n), color[n])

    ax0 = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[3], sharey=ax0)
    df = pd.read_csv('reset_bias_all.csv')
    df = df.head(N)
    _plot(df, ax, 'All layers', color[3])

    # ax = fig.add_subplot(gs[1])
    # ax.set_xlabel('Bias Range')
    plt.tight_layout()
    plt.savefig('fc100-100-10-reset-bias.pdf', format='pdf')
    plt.savefig('fc100-100-10-reset-bias.png', format='png', dpi=dpi)


def shuffle_bad_learner():
    dpi = 200
    fig = plt.figure(figsize=(16, 4), dpi=dpi)
    gs = gridspec.GridSpec(1, 4)

    color = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

    N = 101

    def _plot(df, ax, label, color):
        df.plot(x='range', y='accuracy_mean', ax=ax, label=label,
                c=color)
        x = df['range']
        y = df['accuracy_mean']
        e = df['accuracy_std']
        plt.fill_between(x, y-e, y+e, alpha=0.1, edgecolor='#1B2ACC',
                         facecolor='#089FFF', antialiased=True)
        ax2 = ax.twiny()
        loc_ind = np.array([25, 50, 75])
        ax2_tick_loc = loc_ind
        ax2_tick = ['{0:.1f}%'.format(t * 100)
                    for t in df['count'][ax2_tick_loc]]
        ax2.grid(False)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax2_tick_loc)
        ax2.set_xticklabels(ax2_tick)
        ax.set_xlabel('')
        ax.legend(loc='lower left')

    for n in np.arange(3):
        if n > 0:
            ax0 = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[n], sharey=ax0)
        else:
            ax = fig.add_subplot(gs[n])
        df = pd.read_csv('shuffle_bad_learner_{0}.csv'.format(n))
        df = df.head(N)
        _plot(df, ax, 'Layer {0}'.format(n), color[n])

    ax3 = fig.add_subplot(gs[3], sharey=ax0)
    df = pd.read_csv('shuffle_bad_learner_all.csv')
    df = df.head(N)
    _plot(df, ax3, 'All layers', color[3])

    plt.tight_layout()
    plt.savefig('fc100-100-10-shuffle-weight.pdf', format='pdf')
    plt.savefig('fc100-100-10-shuffle-weight.png', format='png',
                dpi=dpi)


def shade(color, percent):
    C = int(color[1:], base=16)
    R = C >> 16
    G = (C >> 8) & 0x00FF
    B = C & 0x0000FF

    if percent < 0:
        t = 0
        p = -percent
    else:
        t = 255
        p = percent

    return '#{0:06X}'.format(
        (round((t - R) * p) + R) * 0x10000
        + (round((t - G) * p) + G) * 0x100
        + (round((t - B) * p) + B))


def reset_bad_learner():
    dpi = 200
    fig = plt.figure(figsize=(16, 4), dpi=dpi)
    gs = gridspec.GridSpec(1, 4)

    color = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    darker = [shade(c, -0.2) for c in color]

    N = 101

    def _plot(df, ax, label, color):
        df.plot(x='range', y='accuracy_mean', ax=ax, label=label,
                c=color)
        x = df['range']
        y = df['accuracy_mean']
        e = df['accuracy_std']
        plt.fill_between(x, y-e, y+e, alpha=0.1, edgecolor='#1B2ACC',
                         facecolor='#089FFF', antialiased=True)
        ax2 = ax.twiny()
        loc_ind = np.array([25, 50, 75])
        ax2_tick_loc = loc_ind
        ax2_tick = ['{0:.1f}%'.format(t * 100)
                    for t in df['count'][ax2_tick_loc]]
        ax2.grid(False)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax2_tick_loc)
        ax2.set_xticklabels(ax2_tick)
        ax.set_xlabel('')
        ax.legend(loc='lower left')

    for n in np.arange(3):
        if n > 0:
            ax0 = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[n], sharey=ax0)
        else:
            ax = fig.add_subplot(gs[n])
        df = pd.read_csv('reset_bad_learner_{0}_0.0.csv'.format(n))
        df = df.head(N)
        _plot(df, ax, 'U = 0', color[n])
        df = pd.read_csv('reset_bad_learner_{0}_1.0.csv'.format(n))
        df = df.head(N)
        _plot(df, ax, 'U = 1', darker[n])
        ax.text(40, 0.6, 'Layer {0}'.format(n), size='large')

    ax3 = fig.add_subplot(gs[3], sharey=ax0)
    df = pd.read_csv('reset_bad_learner_all_0.0.csv')
    df = df.head(N)
    _plot(df, ax3, 'U = 0', color[3])
    df = pd.read_csv('reset_bad_learner_all_1.0.csv')
    df = df.head(N)
    _plot(df, ax3, 'U = 1', darker[3])
    ax3.text(60, 0.6, 'All layers'.format(n))

    plt.tight_layout()
    plt.savefig('fc100-100-10-reset-weight.pdf', format='pdf')
    plt.savefig('fc100-100-10-reset-weight.png', format='png',
                dpi=dpi)


def useless_weight_pattern():
    dpi = 200
    fig = plt.figure(figsize=(16, 2), dpi=dpi)
    gs = gridspec.GridSpec(1, 8)

    a, b = weights.iloc[0], weights.iloc[-1]
    diff = np.absolute((b - a) / a * 100)

    q = np.percentile(diff, 50)
    bounds = np.arange(0, q, 1.0)
    N = 50
    nframe = bounds.shape[0]

    def _plot(df, ax, n, iframe):
        start = 0
        for i, (l0, l1) in enumerate(pairwise(size)):
            if i <= n:
                break
            else:
                start += l0 * l1 + l1

        im = ax.imshow(np.zeros((l1, l0)), cmap=plt.get_cmap('gray'),
                        animated=True, vmin=0, vmax=255)
        indices = diff <= bounds[iframe]
        ind = np.where(indices[start : (start+l0*l1)])[0]
        frame = np.zeros((l1 * l0,))
        frame[ind] = 255
        frame = frame.reshape(l1, l0)
        im.set_array(frame)
        ax.set_title('N: {0}% Acc {1:.4f}%'.format(
            int(ind.shape[0] / diff.shape[0] * 100),
            df['accuracy_mean'][iframe]))
        return im

    df = []
    ax = []
    for i in np.arange(3):
        df.append(pd.read_csv('reset_bad_learner_{0}_0.0.csv'
                              .format(i)).head(N))
    df.append(pd.read_csv('reset_bad_learner_all_0.0.csv'
                          .format(i)).head(N))
    ax.append(fig.add_subplot(gs[:5]))
    for i in np.arange(3):
        ax.append(fig.add_subplot(gs[5 + i]))

    def _update(iframe):
        im = []
        for i in np.arange(4):
            im.append(_plot(df[i], ax[i], i, iframe))
        return im

    ani = animation.FuncAnimation(fig, _update, frames=nframe,
                                  blit=False)

    plt.tight_layout()
    plt.axis('off')
    ani.save('bad_learner.gif', writer='imagemagick', fps=5, dpi=dpi)


def final_result():
    dpi = 200
    fig = plt.figure(figsize=(8, 2.5), dpi=dpi)
    ax = fig.add_subplot(111)
    df = pd.read_csv('final_result.csv')
    df['nparam'] /= df['nparam'][0]
    df.plot(y='accuracy', x='nparam', ax=ax, legend=None)
    ax.set_xlabel('Compress ratio')
    ax.set_ylabel('Accuracy')

    ax2 = ax.twiny()
    ax.set_xticks(np.linspace(ax.get_xbound()[0], ax.get_xbound()[1], 5))
    ax2.set_xticks(np.linspace(df['bounds'][0], df['bounds'].iloc[-1], 5))

    plt.tight_layout()
    plt.savefig('final_result.pdf', format='pdf')
    plt.savefig('final_result.png', format='png', dpi=dpi)


if __name__ == '__main__':
    # reset_bias()
    # how_much_learned_box()
    # how_much_learned_kde()
    # how_much_learned_kde_absolute()
    # shuffle_bad_learner()
    # reset_bad_learner()
    # useless_weight_pattern()
    final_result()
