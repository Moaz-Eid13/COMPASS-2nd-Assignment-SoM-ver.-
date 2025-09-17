import rebound
import numpy as np
import matplotlib.pyplot as plt

colors = ['red', 'blue', 'green']
T = 200.0
G = 1
dt = 0.001
time = np.arange(0, T, dt)
steps = int(T/dt)
x_values = np.arange(-0.3370767020, -0.3370767020 + 0.05*10, 0.05)

all_simulations = []

for x_0 in x_values:
    sim = rebound.Simulation()
    sim.add(m=1.0, x=x_0, vy=0.9174260238)
    sim.add(m=1.0, x=2.1164029743, vy=-0.0922665014)
    sim.add(m=1.0, x=-1.7793262723, vy=-0.8251595224)

    sim.move_to_com()

    r_history = np.zeros((3, steps, 2))

    for i, t in enumerate(time):
        sim.integrate(t)
        for j, p in enumerate(sim.particles):
            r_history[j, i, 0] = p.x
            r_history[j, i, 1] = p.y
            
    all_simulations.append(r_history)
            
# Figure 1: First 5 simulations
fig1, axes1 = plt.subplots(1, 5, figsize=(20, 4))
fig1.suptitle('Broucke A12 System: Simulations 1-5')

for i in range(5):
    r_history = all_simulations[i]
    for j in range(3):
        axes1[i].plot(r_history[j, :, 0], r_history[j, :, 1], color=colors[j], label=f'Body {j+1}')
        axes1[i].scatter(r_history[j, 0, 0], r_history[j, 0, 1], color='k', marker='o')
    
    axes1[i].set_title(f'x₀ = {x_values[i]:.4f}')
    axes1[i].set_xlabel('x')
    if i == 0:
        axes1[i].set_ylabel('y')
        axes1[i].legend()
    axes1[i].axis('equal')
    axes1[i].grid()

plt.tight_layout()
# plt.show()
plt.savefig('1-5.png', dpi=300)

# # Figure 2: Last 5 simulations
# fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))
# fig2.suptitle('Broucke A12 System: Simulations 6-10')

# for i in range(5):
#     r_history = all_simulations[i + 5]
#     for j in range(3):
#         axes2[i].plot(r_history[j, :, 0], r_history[j, :, 1], color=colors[j], label=f'Body {j+1}')
#         axes2[i].scatter(r_history[j, 0, 0], r_history[j, 0, 1], color='k', marker='o')
    
#     axes2[i].set_title(f'x₀ = {x_values[i + 5]:.4f}')
#     axes2[i].set_xlabel('x')
#     if i == 0:
#         axes2[i].set_ylabel('y')
#         axes2[i].legend()
#     axes2[i].axis('equal')
#     axes2[i].grid()

# plt.tight_layout()
# # plt.show()

# plt.savefig('6-10.png', dpi=300)