import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize

rand_seed=100
N=400
def make_data_normal(data_count=N): #rozklad normalny
    np.random.seed(rand_seed)
    x = np.random.normal(0, 1, data_count)
    dist = lambda z: stats.norm(0, 1).pdf(z)
    return x, dist

def make_data_binormal(data_count=N):   #rozklad binormalny (2 gorki)
    alpha = 0.3
    np.random.seed(rand_seed)
    x = np.concatenate([
        np.random.normal(-1, 2, int(data_count * alpha)),
        np.random.normal(5, 1, int(data_count * (1 - alpha)))
    ])
    dist = lambda z: alpha * stats.norm(-1, 2).pdf(z) + (1 - alpha) * stats.norm(5, 1).pdf(z)
    return x, dist

def kernel(k: str):

    if k not in ['gauss', 'epanechnikov', 'cos', 'trojkat']:
        raise ValueError('niepoprawny/nieznany kernel.')

    def bounded(f): #odcinanie wartosci, ktore nie sa z przedzialu (-1;1)
        def _f(x):
            return f(x) if np.abs(x) <= 1 else 0
        return _f

    if k == 'gauss':
        return lambda u: 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * u * u)
    elif k == 'epanechnikov':
        return bounded(lambda u: (3 / 4 * (1 - u * u)))
    elif k =='cos':
        return bounded(lambda u: np.pi / 4 * np.cos(np.pi / 2 * u))
    elif k == 'trojkat':
        return bounded(lambda u: 1 - np.abs(u))


def kde(data, k=None, h=None, x=None):
    if x is None:
        x = np.linspace(np.min(data), np.max(data), N*10)
    if h is None:
        h = bw_silverman(data)
    if k is None:
        k = kernel('gauss')
    n = len(data)
    kde = np.zeros_like(x)
    for j in range(len(x)):
        for i in range(n):
            kde[j] += k((x[j] - data[i]) / h)
        kde[j] /= n * h
    return kde

data = [('Normal', make_data_normal),('Bimodal (Normal)', make_data_binormal),]
kernels = [('Gauss', kernel('gauss')),('Epanechnikov', kernel('epanechnikov')),('Cos', kernel('cos')),('Trojkat', kernel('trojkat'))]

#rozne zasady generowania parametru h (bw - bandwidth)

def bw_scott(data: np.ndarray):
    std_dev = np.std(data, axis=0, ddof=1)
    n = len(data)
    return 3.49 * std_dev * n ** (-0.333)

def bw_silverman(data: np.ndarray):
    def _select_sigma(x):
        normalizer = 1.349
        iqr = (stats.scoreatpercentile(x, 75) - stats.scoreatpercentile(x, 25)) / normalizer
        std_dev = np.std(x, axis=0, ddof=1)
        return np.minimum(std_dev, iqr) if iqr > 0 else std_dev
    sigma = _select_sigma(data)
    n = len(data)
    return 0.9 * sigma * n ** (-0.2)

def bw_mlcv(data: np.ndarray, k):
    n = len(data)
    x = np.linspace(np.min(data), np.max(data), n)
    def mlcv(h):
        fj = np.zeros(n)
        for j in range(n):
            for i in range(n):
                if i == j: continue
                fj[j] += k((x[j] - data[i]) / h)
            fj[j] /= (n - 1) * h
        return -np.mean(np.log(fj[fj > 0]))
    h = optimize.minimize(mlcv, 1)
    if np.abs(h.x[0]) > 10:
        return bw_scott(data)
    return h.x[0]

bw_algorithms = [('Scott', bw_scott),('Silverman', bw_silverman),('MLCV', bw_mlcv),]

def run_kde(ax, data, kernel):
    x, dist = data[1]()
    x_plot = np.linspace(np.min(x) * 1.05, np.max(x) * 1.05, N*10)
    ax.grid(True)
    ax.fill_between(x_plot, dist(x_plot), fc='silver', alpha=0.5)
    ax.plot(x, np.full_like(x, -0.02), '|k', markeredgewidth=1)
    ax.hist(x, density=True, alpha=0.2, bins=int(N/10), rwidth=0.9)
    for bw in bw_algorithms:
        if bw[0] == 'MLCV':
            h = bw[1](x, kernel[1])
        else:
            h = bw[1](x)
        x_kde = kde(x, kernel[1], h=h, x=x_plot)
        ax.plot(x_plot, x_kde, linewidth=1, label='$h_{\mathrm{' + bw[0] + '}} = ' + str(round(h, 5)) + '$')
    ax.legend(loc='best', fontsize='small')
    ax.set_title(f'{data[0]}, {kernel[0]}')

fig, axs = plt.subplots(len(data), len(kernels), figsize=(16, 12))

for i, d in enumerate(data):
    for j, k in enumerate(kernels):
        run_kde(axs[i, j], d, k)

fig.tight_layout()
plt.show()
fig.savefig('porownanie_2.pdf')