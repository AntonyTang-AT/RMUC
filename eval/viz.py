import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

STATE_COLORS = {'def': '#1e88e5', 'hitbuilding': '#e53935', 'cruise': '#43a047'}

def plot_state_strip(t_axis, labels, path):
    fig, ax = plt.subplots(figsize=(14, 2.5))
    for i, (t, lbl) in enumerate(zip(t_axis, labels)):
        ax.barh(0, 1, left=t, height=1.0, color=STATE_COLORS[lbl], edgecolor='none')
    ax.set_yticks([])
    ax.set_xlabel('time (s)')
    ax.set_xlim(0, max(t_axis) + 1 if t_axis else 1)
    ax.set_title('sentry state over time')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in STATE_COLORS.items()], loc='upper right')
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_heatmap(xs, ys, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs, ys, s=2, alpha=0.3)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.axis('equal'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path); plt.close()
