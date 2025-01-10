import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.integrate import solve_ivp


sigma = 10
rho = 28
beta = 8 / 3


def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    d_vec = [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    return d_vec


init_state = [0, 1, 1]
time_points = np.linspace(0, 40, 1001)

sol = solve_ivp(
    lorenz_system, [0, 40], init_state, args=(sigma, rho, beta), t_eval=time_points
)
x, y, z = sol.y


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))

def update_lines(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])
    return line

data = np.array([x, y, z])
line, = ax.plot(data[0], data[1], data[2], lw=0.5)

ani = FuncAnimation(fig, update_lines, frames=len(time_points), fargs=(data, line), interval=30, blit=False)

plt.show()