from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
import os
import tempfile
from datetime import datetime

app = Flask(__name__)

if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('static/generated_plots'):
    os.makedirs('static/generated_plots')

def acceleration(r, m, G=1):
    acc = np.zeros((3, 2), dtype=float)
    for i in range(3):
        for j in range(3):
            if i != j:
                r_vec = r[j] - r[i]
                dist = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
                if dist > 0:
                    acc[i] += G * m[j] * r_vec / dist**3
    return acc

def total_energy(r, v, m, G=1):
    T = 0.5 * np.sum(m[:, None] * np.sum(v * v, axis=1, keepdims=True))
    U = 0.0
    N = len(m)
    for i in range(N):
        for j in range(i + 1, N):
            U -= G * m[i] * m[j] / np.linalg.norm(r[i] - r[j])
    return U + T

def total_angular_momentum(r, v, m):
    cross_terms = r[:,0] * v[:,1] - r[:,1] * v[:,0]
    L = np.sum(m * cross_terms)
    return L

def forward_euler_integration(r_0, v_0, m, T, dt, G=1):
    steps = int(T/dt)
    r_history = np.zeros((3, steps, 2), dtype=float)
    E_history = np.zeros(steps)
    L_history = np.zeros(steps)
    
    r = r_0.copy()
    v = v_0.copy()
    a = acceleration(r, m, G)
    
    for i in range(steps):
        r = r + v * dt
        v = v + a * dt
        a = acceleration(r, m, G)
        r_history[:, i, :] = r
        E_history[i] = total_energy(r, v, m, G)
        L_history[i] = total_angular_momentum(r, v, m)
        
    return r_history, E_history, L_history

def rk2_integration(r_0, v_0, m, T, dt, G=1):
    steps = int(T/dt)
    r_history = np.zeros((3, steps, 2), dtype=float)
    E_history = np.zeros(steps)
    L_history = np.zeros(steps)
    
    r = r_0.copy()
    v = v_0.copy()
    a = acceleration(r, m, G)
    
    for i in range(steps):
        r_mid = r + 0.5 * dt * v
        v_mid = v + 0.5 * dt * a
        a_mid = acceleration(r_mid, m, G)
        r += dt * v_mid
        v += dt * a_mid
        a = a_mid
        r_history[:, i, :] = r
        E_history[i] = total_energy(r, v, m, G)
        L_history[i] = total_angular_momentum(r, v, m)
        
    return r_history, E_history, L_history

def velocity_verlet_integration(r_0, v_0, m, T, dt, G=1):
    steps = int(T/dt)
    r_history = np.zeros((3, steps, 2), dtype=float)
    E_history = np.zeros(steps)
    L_history = np.zeros(steps)
    
    r = r_0.copy()
    v = v_0.copy()
    a = acceleration(r, m, G)
    
    for i in range(steps):
        r += v * dt + 0.5 * a * dt**2
        a_new = acceleration(r, m, G)
        v += 0.5 * (a + a_new) * dt
        a = a_new
        r_history[:, i, :] = r
        E_history[i] = total_energy(r, v, m, G)
        L_history[i] = total_angular_momentum(r, v, m)
        
    return r_history, E_history, L_history

def generate_plots(r_history, E_history, L_history, T, dt, method_name):
    steps = len(E_history)
    time = np.arange(steps) * dt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['blue', 'red', 'green']
    for i in range(3):
        axes[0].plot(r_history[i, :, 0], r_history[i, :, 1], label=f'Body {i+1}', color=colors[i])
        axes[0].scatter(r_history[i, 0, 0], r_history[i, 0, 1], 
                        marker='o', c='black', s=50, zorder=5)
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'3-Body Orbits ({method_name}, dt = {dt})')
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot energy
    axes[1].plot(time, E_history, label="Total Energy", color='red', linewidth=2)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Energy")
    axes[1].set_title(f"Energy Conservation ({method_name})")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot angular momentum
    axes[2].plot(time, L_history, label="Angular Momentum", color='blue', linewidth=2)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Angular Momentum")
    axes[2].set_title(f"Angular Momentum Conservation ({method_name})")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save plot using absolute path based on app.root_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"plot_{method_name.lower().replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join(app.root_path, 'static', 'generated_plots', filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate')
def simulate():
    return render_template('simulate.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    try:
        data = request.json
        
        # Extract parameters
        method = data.get('method', 'velocity_verlet')
        dt = float(data.get('dt', 0.01))
        T = float(data.get('T', 50))
        G = float(data.get('G', 1.0))
        
        # Extract masses
        masses = [
            float(data.get('m1', 1.0)),
            float(data.get('m2', 1.0)),
            float(data.get('m3', 1.0))
        ]
        m = np.array(masses)
        
        # Extract initial positions
        r_0 = np.array([
            [float(data.get('r1x', -1.0)), float(data.get('r1y', 0.0))],
            [float(data.get('r2x', 1.0)), float(data.get('r2y', 0.0))],
            [float(data.get('r3x', 0.0)), float(data.get('r3y', 0.0))]
        ])
        
        # Extract initial velocities
        v_0 = np.array([
            [float(data.get('v1x', 0.347111)), float(data.get('v1y', 0.532728))],
            [float(data.get('v2x', 0.347111)), float(data.get('v2y', 0.532728))],
            [float(data.get('v3x', -0.694222)), float(data.get('v3y', -1.065456))]
        ])
        
        # Center of mass correction
        p = m[0] * v_0[0] + m[1] * v_0[1] + m[2] * v_0[2]
        v_com = p / np.sum(m)
        v_0 -= v_com
        
        # Choose integration method
        method_functions = {
            'forward_euler': forward_euler_integration,
            'rk2': rk2_integration,
            'velocity_verlet': velocity_verlet_integration
        }
        
        method_names = {
            'forward_euler': 'Forward Euler',
            'rk2': 'Runge-Kutta 2',
            'velocity_verlet': 'Velocity Verlet'
        }
        
        if method not in method_functions:
            return jsonify({'error': 'Invalid integration method'}), 400
        
        # Run simulation
        integration_func = method_functions[method]
        r_history, E_history, L_history = integration_func(r_0, v_0, m, T, dt, G)
        
        # Generate plot
        plot_filename = generate_plots(r_history, E_history, L_history, T, dt, method_names[method])
        
        # Calculate some statistics
        energy_drift = abs(E_history[-1] - E_history[0]) / abs(E_history[0]) * 100
        momentum_drift = abs(L_history[-1] - L_history[0])

        
        return jsonify({
            'success': True,
            'plot_url': f'/static/generated_plots/{plot_filename}',
            'statistics': {
                'initial_energy': float(E_history[0]),
                'final_energy': float(E_history[-1]),
                'energy_drift_percent': float(energy_drift),
                'initial_momentum': float(L_history[0]),
                'final_momentum': float(L_history[-1]),
                'momentum_drift_percent': float(momentum_drift),
                'simulation_time': float(T),
                'time_step': float(dt),
                'total_steps': len(E_history)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preset/<preset_name>')
def load_preset(preset_name):
    """Load predefined initial conditions"""
    presets = {
        'goggles': {
            'r1x': -1.0, 'r1y': 0.0,
            'r2x': 1.0, 'r2y': 0.0,
            'r3x': 0.0, 'r3y': 0.0,
            'v1x': 0.083300, 'v1y': 0.127889,
            'v2x': 0.083300, 'v2y': 0.127889,
            'v3x': -0.1666, 'v3y': -0.255778,
            'm1': 1.0, 'm2': 1.0, 'm3': 1.0,
            'T': 10.466818, 'dt': 0.0001, 'G': 1.0
        },
        'moth1': {   
            'r1x': -1, 'r1y': 0,
            'r2x': 1, 'r2y': 0,
            'r3x': 0, 'r3y': 0,
            'v1x': 0.464445, 'v1y': 0.396060,
            'v2x': 0.464445, 'v2y': 0.396060,
            'v3x': -0.92889, 'v3y': -0.79212,
            'm1': 1.0, 'm2': 1.0, 'm3': 1.0,
            'T': 14.893911, 'dt': 0.0001, 'G': 1.0
        },
        'dragonfly': {
            'r1x': -1.0, 'r1y': 0.0,
            'r2x': 1.0, 'r2y': 0.0,
            'r3x': 0.0, 'r3y': 0.0,
            'v1x': 0.080584, 'v1y': 0.588836,
            'v2x': 0.080584, 'v2y': 0.588836,
            'v3x': -0.161168, 'v3y': -1.177672,
            'm1': 1.0, 'm2': 1.0, 'm3': 1.0,
            'T': 21.270975, 'dt': 0.0001, 'G': 1.0
        }
    }

    key = preset_name.lower()
    if key in presets:
        return jsonify(presets[key])
    else:
        return jsonify({'error': 'Preset not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)