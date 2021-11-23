import numpy as np
import tensorflow as tf
from pathlib import Path
import math
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
from argparse import Namespace, ArgumentParser
import gpflow
from gpflow.utilities import print_summary
from gpflow import default_float

import sys

from pargpfsgp import ParGpfsGp
from model import StateSpaceGP
import bayesian_algorithm_execution.neatplot as neatplot

from bayesian_algorithm_execution.bax.models.gpfs_gp import GpfsGp
from bayesian_algorithm_execution.bax.alg.algorithms import TopK
from bayesian_algorithm_execution.bax.models.pgps.pssgp.kernels.matern import Matern52
from bayesian_algorithm_execution.bax.models.pgps.pssgp.kernels.rbf import RBF
from bayesian_algorithm_execution.bax.acq.acquisition import (
    BaxAcqFunction, UsBaxAcqFunction, EigfBaxAcqFunction, RandBaxAcqFunction
)
from bayesian_algorithm_execution.bax.acq.acqoptimize import AcqOptimizer

def unif_random_sample_domain(domain, n=1):
    """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
    list_of_arr_per_dim = [np.random.uniform(dom[0], dom[1], n) for dom in domain]
    list_of_list_per_sample = [list(l) for l in np.array(list_of_arr_per_dim).T]
    return list_of_list_per_sample

print(tf.__version__)
print(tf.config.list_physical_devices())

neatplot.set_style("fonts")
neatplot.update_rc('font.size', 20)
plt.rc('text', usetex=False)

seed = 12
np.random.seed(seed)
tf.random.set_seed(seed)
n_iter = 20



f = lambda x: math.sin(x*math.pi) + math.sin(2 * math.pi * x) + math.sin(3 * math.pi * x)

n_dim = 1
domain = [[1, 10]] * n_dim
dtype = default_float()

len_path = 150

X = unif_random_sample_domain(domain, len_path)
X = np.asarray(X)
X = X.reshape(-1,n_dim)

Y = [f(x) for x in X]


Y = np.asarray(Y)

k = 10

algo = TopK({"x_path": X, "k": k})

algo_gt = TopK({"x_path": X, "k": k, "name": "groundtruth"})
exepath_gt, output_gt = algo_gt.run_algorithm_on_f(f)
print(f"Algorithm ground truth output is:\n{output_gt}")

data = Namespace()
data.x = unif_random_sample_domain(domain, 1)
data.x = np.asarray(data.x)
data.x = data.x.reshape(-1,1)
data.y = [f(x) for x in data.x]

data.y = np.asarray(data.y)


gp_params = {"ls": 2.5, "alpha": 20.0, "sigma": 1e-2, "n_dimx": n_dim}
modelclass = ParGpfsGp

acqfn_params = {"acq_str": "exe", "n_path": 100, "crop": True}
acq_cls = BaxAcqFunction

n_acqopt = 1

results_dir = Path("parallel_correct_results")
results_dir.mkdir(parents=True, exist_ok=True)

img_dir = results_dir /  f'images'
img_dir.mkdir(parents=True, exist_ok=True)

for i in range(n_iter):
    # Set model
    model = modelclass(gp_params, data)

    # Set and optimize acquisition function
    acqfn = acq_cls(acqfn_params, model, algo)
    x_test = unif_random_sample_domain(domain, n=n_acqopt)
    acqopt = AcqOptimizer({"x_batch": x_test})
    x_next = acqopt.optimize(acqfn)

    # Print iter info
    print(f"Acqopt x_next = {x_next}")
    print(f"output_list[0] = {acqfn.output_list[0]}")
    print(f"Finished iter i = {i}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    # -- plot function contour
    #grid = 0.1
    #xpts = np.arange(domain[0][0], domain[0][1], grid)
    #ypts = np.arange(domain[1][0], domain[1][1], grid)
    #X, Y = np.meshgrid(xpts, ypts)
    #Z = f_vec(X, Y)
    #ax.contour(X, Y, Z, 20, cmap=cm.Greens_r, zorder=0)
    # -- plot top_k
    
    topk_arr = np.array(output_gt.x)
    y_arr = [f(x[0]) for x in topk_arr]

    ax.plot(
        topk_arr[:, 0],
        y_arr,
        '*',
        marker='D',
        markersize=15,
        color='c',
        markeredgecolor='black',
        markeredgewidth=0.05,
        zorder=1
    )
    # -- plot x_path
    #x_path_arr = np.array(x_path)
    #ax.plot(x_path_arr[:, 0], x_path_arr[:, 1], '.', color='#C0C0C0', markersize=8)
    ax.plot(X, Y, 'o', color='k', markersize=8)
    # -- plot observations
    for x in data.x:
        ax.scatter(x[0], f(x[0]), color='r', s=80)
    # -- plot x_next
    ax.scatter(x_next[0], f(x_next[0]), color='g', s=80, zorder=10)
    # -- plot estimated output
    for out in acqfn.output_list:
        out_arr = np.array(out.x)
        y_out_arr = [f(x[0]) for x in out_arr]
        ax.plot(
            out_arr[:, 0], y_out_arr, 's', markersize=8, color='m', alpha=0.02
        )
    # -- lims, labels, titles, etc
    #ax.set(xlim=domain[0], ylim=domain[1])
    ax.set_title("Parallel InfoBAX with Top-$k$ Algorithm")

    # Save plot
    img_path = img_dir / f'topk_{i}'
    neatplot.save_figure(str(img_path), 'pdf')
    """
    # Query function, update data

    
    y_next = f(x_next[0])
    data.x = np.append(data.x, [x_next])
    data.y = np.append(data.y, [y_next])
    data.x = data.x.reshape(-1,1)
    
    #data.x = data.x.tolist()
    #data.y = data.y.reshape(-1,1)
    """
    #data.x.append(x_next)
    #data.y.append(y_next)

#results.data = data

# Pickle results
file_str = f"topk.pkl"
with open(results_dir / file_str, "wb") as handle:
    #pickle.dump(results, handle)
    print(f"Saved results file: {results_dir}/{file_str}")
    
