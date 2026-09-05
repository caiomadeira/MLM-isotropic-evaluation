import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = "results/figs"

def _save(title):
    os.makedirs(OUTDIR, exist_ok=True)
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
    path = os.path.join(OUTDIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def plot_2d_projection(w, title):
    _, eigvecs = np.linalg.eigh(w.T @ w)
    projection = w @ eigvecs[:, [-1, -2]]
    plt.figure(figsize=(8,6))

    graph_limit = np.max(np.abs(projection)) * 1.05

    plt.scatter(projection[:, 0], projection[:, 1], s=0.1, color="tab:blue")
    plt.xlim(-graph_limit, graph_limit)
    plt.ylim(-graph_limit, graph_limit)
    plt.xticks([-graph_limit, 0, graph_limit], [f"{-graph_limit:.4f}", "0.0000", f"{graph_limit:.4f}"])
    plt.yticks([-graph_limit, 0, graph_limit], [f"{-graph_limit:.4f}", "0.0000", f"{graph_limit:.4f}"])

    plt.title(title)
    return _save(title)

def plot_hexbin_projection2(w, title):
    eigvals, eigvecs = np.linalg.eigh(w.T @ w)
    projection = w @ eigvecs[:, [-1, -2]]
    var = eigvals[::-1] / eigvals.sum()
    lim = np.percentile(np.abs(projection), 99.5) * 1.15
    plt.figure(figsize=(7, 7))
    plt.hexbin(projection[:,0], projection[:,1], gridsize=90, cmap="Blues", bins="log", mincnt=1)
    plt.plot(0, 0, "+", color="red", ms=14, mew=2)
    plt.xlim(-lim, lim); plt.ylim(-lim, lim); plt.gca().set_aspect("equal")
    #plt.title(f"{title}\n2 axis (d=2) covers {100*var[:2].sum():.1f}% of variance")
    print(f"{title}\n2 axis (d=2) covers {100*var[:2].sum():.1f}% of variance")
    plt.title(title, fontsize=22)
    _save(title + "_hexbin")
    return float(var[:2].sum())