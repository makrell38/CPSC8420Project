import copy
from argparse import Namespace, ArgumentParser
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

from non_Dom import nonDom
from bax.models.simple_gp import SimpleGp
from bax.models.gpfs_gp import MultiGpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition import (
    BaxAcqFunction, UsBaxAcqFunction, EigfBaxAcqFunction, RandBaxAcqFunction
)
from bax.acq.acqoptimize import AcqOptimizer
from bax.acq.visualize import AcqViz1D
from bax.util.domain_util import unif_random_sample_domain

import neatplot
neatplot.set_style("fonts")
neatplot.update_rc('font.size', 20)

plt.rc('text', usetex=False)

seed = 12
n_init = 1
n_iter = 20


print(f"*[INFO] Seed: {seed}")
np.random.seed(seed)
tf.random.set_seed(seed)

f = lambda x: [np.sin(x[0])*x[1], np.cos(x[1])*x[0]]

# Set algorithm  details
n_dim = 2
domain = [[0, 1], [0, np.pi*.5]] 
len_path = 150

x_path = unif_random_sample_domain(domain, len_path)
algo = nonDom({"x_path": x_path}, 2)

# Get ground truth algorithm output
algo_gt = nonDom({"x_path": x_path, "name": "groundtruth"}, 2)
exepath_gt, output_gt = algo_gt.run_algorithm_on_f(f)
print(f"Algorithm ground truth output is:\n{output_gt}")

# Set data for model
data = Namespace()
data.x = unif_random_sample_domain(domain, n_init)
data.y = [f(x) for x in data.x]

gp_params = {"ls": 2.5, "alpha": 20.0, "sigma": 1e-2, "n_dimx": n_dim}
modelclass = MultiGpfsGp

# Set acquisition details
acqfn_params = {"acq_str": "exe", "n_path": 100, "crop": True}
#acq_cls = BaxAcqFunction
acq_cls = EigfBaxAcqFunction

# Set acqopt details
n_acqopt = 1500

# Set up results directory
results_dir = Path("nonDom_pareto")
results_dir.mkdir(parents=True, exist_ok=True)

# Set up img directory
img_dir = results_dir /  f'images_{seed}'
img_dir.mkdir(parents=True, exist_ok=True)

"""
fig, ax = plt.subplots(figsize=(6, 6))

x_arr = np.array(x_path)


ax.plot(x_arr[:, 0], x_arr[:, 1], 'o', color='k', markersize=3)

x_output_arr = np.array(output_gt)
ax.plot(
        x_output_arr[:, 0],
        x_output_arr[:, 1],
        '*',
        marker='D',
        markersize=15,
        color='c',
        markeredgecolor='black',
        markeredgewidth=0.05,
        zorder=1
    )

#ax.set(xlim=[0,1.5], ylim=[0,.6])
ax.set_title("Pareto Frontier")
ax.set_xlabel("X[1]")
ax.set_ylabel("X[2]")

img_path = img_dir / f'nondom_X'
neatplot.save_figure(str(img_path), 'pdf')
"""
# Run BAX loop
for i in range(n_iter):
    # Set model
    model = modelclass({"name": "MultiGpfsGp", "n_dimy": 2, "gp_params":gp_params}, data)

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
    grid = 0.1

    # -- plot top_k
    nondom_arr = np.array(output_gt)
    ax.plot(
        nondom_arr[:, 0],
        nondom_arr[:, 1],
        '*',
        marker='D',
        markersize=15,
        color='c',
        markeredgecolor='black',
        markeredgewidth=0.05,
        zorder=1
    )

    # -- plot x_path
    x_path_arr = np.array(x_path)
    ax.plot(x_path_arr[:, 0], x_path_arr[:, 1], 'o', color='k', markersize=3)
    # -- plot observations
    for x in data.x:
        ax.scatter(x[0], x[1], color='r', s=80)
    # -- plot x_next
    ax.scatter(x_next[0], x_next[1], color='g', s=80, zorder=10)
    # -- plot estimated output
    for out in acqfn.output_list:
        out_arr = np.array(out.x)
        ax.plot(
            out_arr[:,0], out_arr[:,1], 's', markersize=8, color='m', alpha=0.02
        )
    # -- lims, labels, titles, etc
    #ax.set(xlim=domain[0], ylim=domain[1])
    ax.set_title("BAX with Multi-objective Optimization Algorithm")
    ax.set_xlabel("X[1]")
    ax.set_ylabel("X[2]")

    # Save plot
    img_path = img_dir / f'nondom_{i}'
    neatplot.save_figure(str(img_path), 'pdf')

    # Query function, update data
    y_next = (f(x_next))
    data.x.append(x_next)
    data.y.append(y_next)


file_str = f"nondom_{seed}.pkl"
with open(results_dir / file_str, "wb") as handle:
    print(f"Saved results file: {results_dir}/{file_str}")
