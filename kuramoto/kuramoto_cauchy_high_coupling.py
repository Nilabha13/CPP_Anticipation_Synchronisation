import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


def simulate_kuramoto(N, K, freq_dist, init_states, time_points):
    """
    N: number of oscillators
    K: coupling strength
    freq_dist: frequency distribution
    init_states: initial states of oscillators
    time_points: time points to simulate
    """
    frequencies = np.array([freq_dist() for _ in range(N)])

    def kuramoto(t, theta):
        dtheta = frequencies - (K/N) * np.sum(np.sin(theta[:, None] - theta[None, :]), axis=1)
        return dtheta

    sol = solve_ivp(kuramoto, (time_points[0], time_points[-1]), init_states, t_eval=time_points)
    return sol.y


time_points = np.linspace(0, 100, 1000)
N = 100
K = 10
freq_dist = lambda: np.random.standard_cauchy()
init_states = np.random.uniform(-np.pi, np.pi, N)
states = simulate_kuramoto(N, K, freq_dist, init_states, time_points)

radii = 100 * np.random.uniform(0.9, 1.1, N)

fig, ax = plt.subplots()
ax.set_xlim([-150, 150])
ax.set_ylim([-150, 150])
ax.axis('equal')

scat = ax.scatter(radii * np.cos(states[:, 0]), radii * np.sin(states[:, 0]), s=10)

SPEEDUP = 1

def update(frame):
    scat.set_offsets(np.c_[radii * np.cos(states[:, frame*SPEEDUP]), radii * np.sin(states[:, frame*SPEEDUP])])
    return scat

ani = FuncAnimation(fig, update, frames=len(time_points)//SPEEDUP, interval=1, repeat=False)
plt.show()