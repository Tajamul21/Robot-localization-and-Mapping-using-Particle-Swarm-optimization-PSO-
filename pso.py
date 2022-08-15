from __future__ import division
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


def generate_random(x, y):
    A = (np.random.rand(x, y) * 200) - 100  # Generating random matrix in [-100,100]
    return A


# ---- DEFINE CONSTANTS ---------------------------------------------------------+

n_a = 4  # Number of anchor nodes
n_t = 1  # Number of target nodes
n_p = 200  # Number of particles for PSO
n_i = 100  # Number of iterations
bounds_min = -100  # Maximum coordinates of x and y
bounds_max = 100  # Minimum coordinates of x and y
anc_n = np.array([[0, 100], [0, -100], [100, 0], [-100, 0]])  # Anchor or reference node
tar_n = generate_random(n_t, 2)  # Target node


# --- COST FUNCTION ------------------------------------------------------------+

# Function we are attempting to optimize (minimize)
def func1(x):
    total = 0
    for k in range(0, n_t):
        for j in range(0, n_a):
            for i in range(0, 2):
                total += abs((anc_n[j][i] - x[i]) ** 2 - (anc_n[j][i] - tar_n[k][i]) ** 2)
    return total


# ---------- MAIN --------------------------------------------------------------+

class Particle:
    def __init__(self, x0):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = 999999  # best error individual
        self.err_i = -1  # error individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1.3  # cognative constant
        c2 = 1.5  # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()
            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds_max:
                self.position_i[i] = bounds_max

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds_min:
                self.position_i[i] = bounds_min


class PSO():
    def __init__(self, costFunc, x0, num_particles, maxiter):
        global num_dimensions

        num_dimensions = len(x0)
        err_best_g = 999999  # best error for group
        pos_best_g = []  # best position for group

        # establish the swarm
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i = 0
        while i < maxiter:
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                swarm[j].evaluate(costFunc)
                if swarm[j].err_i < err_best_g:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position()
            i += 1

        # print final results
        print('FINAL:')
        print(tar_n[0])
        print(pos_best_g)
        print(err_best_g)

        data = anc_n
        x, y = data.T
        plt.scatter(x, y, color="blue", s=100, label='Anchor nodes')

        data1 = tar_n
        x1, y1 = data1.T
        plt.scatter(x1, y1, color="red", s=200, label='Target nodes', marker='+')

        data2 = np.matrix([[pos_best_g[0], pos_best_g[1]]])
        x2, y2 = data2.T
        plt.scatter(pos_best_g[0], pos_best_g[1], color="green", s=150, label='Target estimation', marker='x')

        plt.legend(loc='upper right')

        plt.show()


if __name__ == "__PSO__":
    main()

# --- RUN ----------------------------------------------------------------------+

initial = generate_random(n_t, 2)[0]  # initial starting location [x1,x2...]
PSO(func1, initial, num_particles=n_p, maxiter=n_i)