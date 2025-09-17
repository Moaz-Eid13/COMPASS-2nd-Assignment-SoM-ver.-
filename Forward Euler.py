import numpy as np
import matplotlib.pyplot as plt

G = 1
T = 200
dt = np.array([0.1, 0.001])
steps = np.array([int(T/dt[0]), int(T/dt[1])])
p1, p2 = 0.347111, 0.532728
m = np.full(3, 1.0)
r_0 = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
r_history1 = np.zeros((3, steps[0], 2), dtype=float)
r_history2 = np.zeros((3, steps[1], 2), dtype=float)
v_0 = np.array([[p1, p2], [p1, p2], [-2*p1, -2*p2]])
def acceleration(r, m):
    acc = np.zeros((3, 2), dtype=float)
    for i in range(3):
        for j in range(3):
            if i != j:
                r_vec = r[j] - r[i]
                dist = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
                if dist > 0:
                    acc[i] += G * m[j] * r_vec / dist**3
    return acc
p = m[0] * v_0[0] + m[1] * v_0[1] + m[2] * v_0[2]
v_com = p / np.sum(m)
v_0 -= v_com
E_history1 = np.zeros(steps[0])
L_history1 = np.zeros(steps[0])
E_history2 = np.zeros(steps[1])
L_history2 = np.zeros(steps[1])

def total_energy(r, v, m):
    T = 0.5 * np.sum(m[:, None] * np.sum(v*v, axis=1, keepdims=True))
    U = 0.0
    N = len(m)
    for i in range(N):
        for j in range(i+1, N):
            U -= G * m[i] * m[j] / np.linalg.norm(r[i] - r[j])
    return T + U

def total_angular_momentum(r, v, m):
    cross_terms = r[:,0]*v[:,1] - r[:,1]*v[:,0]
    L = np.sum(m * cross_terms)
    return L

r = r_0.copy()
v = v_0.copy()
a = acceleration(r, m)

# --- Forward Euler integration --- [0]
for i in range(steps[0]):
    r = r + v * dt[0]
    v = v + a * dt[0]
    a = acceleration(r, m)
    r_history1[:, i, :] = r
    E_history1[i] = total_energy(r, v, m)
    L_history1[i] = total_angular_momentum(r, v, m)

r = r_0.copy()
v = v_0.copy()
a = acceleration(r, m)

# --- Forward Euler integration --- [1]
for i in range(steps[1]):
    r = r + v * dt[1]
    v = v + a * dt[1]
    a = acceleration(r, m)
    r_history2[:, i, :] = r
    E_history2[i] = total_energy(r, v, m)
    L_history2[i] = total_angular_momentum(r, v, m)

time = [np.arange(steps[i]) * dt[i] for i in range(len(dt))]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i in range(3):
    axes[0, 0].plot(r_history1[i, :, 0], r_history1[i, :, 1], label=f'Body {i+1}')
    axes[0, 0].scatter(r_history1[i, 0, 0], r_history1[i, 0, 1], marker='o', c='k')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].set_title('3-Body Orbits (dt = 0.1)')
axes[0, 0].legend()
axes[0, 0].axis('equal')
axes[0, 0].grid()

axes[0, 1].plot(time[0], E_history1, label="Total Energy", color='red')
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Energy")
axes[0, 1].set_title("Energy Conservation (dt = 0.1)")
axes[0, 1].grid()
axes[0, 1].legend()

axes[0, 2].plot(time[0], L_history1, label="Angular Momentum", color='blue')
axes[0, 2].set_xlabel("Time")
axes[0, 2].set_ylabel("Angular Momentum")
axes[0, 2].set_title("Angular Momentum Conservation (dt = 0.1)")
axes[0, 2].grid()
axes[0, 2].legend()

for i in range(3):
    axes[1, 0].plot(r_history2[i, :, 0], r_history2[i, :, 1], label=f'Body {i+1}')
    axes[1, 0].scatter(r_history2[i, 0, 0], r_history2[i, 0, 1], marker='o', c='k')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('3-Body Orbits (dt = 0.001)')
axes[1, 0].legend()
axes[1, 0].axis('equal')
axes[1, 0].grid()

axes[1, 1].plot(time[1], E_history2, label="Total Energy", color='red')
axes[1, 1].set_xlabel("Time")
axes[1, 1].set_ylabel("Energy")
axes[1, 1].set_title("Energy Conservation (dt = 0.001)")
axes[1, 1].grid()
axes[1, 1].legend()

axes[1, 2].plot(time[1], L_history2, label="Angular Momentum", color='blue')
axes[1, 2].set_xlabel("Time")
axes[1, 2].set_ylabel("Angular Momentum")
axes[1, 2].set_title("Angular Momentum Conservation (dt = 0.001)")
axes[1, 2].grid()
axes[1, 2].legend()

plt.tight_layout()
plt.show()