from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_env.simulation import load_scenario

SCENARIO_JSON = 'scenario991.json'
VEHICLE_ID = 'perf49_02'

scenario_file = Path(__file__).resolve().parent / 'demo' / SCENARIO_JSON
scenario = load_scenario(scenario_file)
if VEHICLE_ID not in scenario.routes:
    raise ValueError(f'Vehicle {VEHICLE_ID!r} not found in {scenario_file}')
route = np.asarray(scenario.routes[VEHICLE_ID]['points'], dtype=float)[:, :2]
if len(route) < 2:
    raise ValueError('Long route needs at least two points')
lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
arc = np.r_[0.0, np.cumsum(lengths)]
total = float(arc[-1])


def point_at(s):
    s = float(np.clip(s, 0.0, total))
    i = min(int(np.searchsorted(arc, s, side='right') - 1), len(route) - 2)
    fraction = (s - arc[i]) / max(float(lengths[i]), 1e-12)
    return route[i] + fraction * (route[i + 1] - route[i])


def draw(ax, end_s, arrow_count, arrow_length):
    visible = np.vstack((route[arc < end_s], point_at(end_s)))
    ax.plot(visible[:, 0], visible[:, 1], color='tab:blue', lw=1.8)
    for s in np.linspace(0.0, max(0.0, end_s - arrow_length), arrow_count):
        ax.annotate('', xy=point_at(s + arrow_length), xytext=point_at(s),
                    arrowprops={'arrowstyle': '->', 'color': 'tab:red', 'lw': 1.3})
    ax.scatter(*route[0], color='tab:green', s=50, zorder=4)
    ax.annotate('start', route[0], xytext=(5, 5), textcoords='offset points')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set(xlabel='x (m)', ylabel='y (m)')
    ax.grid(alpha=0.3)


fig, (full_ax, near_ax) = plt.subplots(1, 2, figsize=(15, 7), layout='constrained')
draw(full_ax, total, 22, min(15.0, total / 30.0))
full_ax.scatter(*route[-1], color='tab:orange', s=50, zorder=4)
full_ax.annotate('destination', route[-1], xytext=(5, 5), textcoords='offset points')
full_ax.set_title(f'Complete long route ({total:.1f} m)')

near = min(120.0, total)
draw(near_ax, near, 16, min(4.0, near / 20.0))
for s in (50.0, 80.0):
    if s <= total:
        point = point_at(s)
        near_ax.scatter(*point, s=50, zorder=5)
        near_ax.annotate(f's={s:.0f} m', point, xytext=(6, 6),
                         textcoords='offset points')
near_ax.set_title(f'First {near:.0f} m; red arrows show direction')
fig.suptitle(f'{SCENARIO_JSON} | {VEHICLE_ID}')
plt.show()
