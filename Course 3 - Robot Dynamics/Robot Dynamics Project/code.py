import numpy as np
import modern_robotics as mr
import csv

# Define transformation matrices for each link (Mlist)
M01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])
M12 = np.array([[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]])
M23 = np.array([[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]])
M34 = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]])
M45 = np.array([[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]])
M56 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]])
M67 = np.array([[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]])
Mlist = [M01, M12, M23, M34, M45, M56, M67]

# Inertia matrices for each link (Glist)
G1 = np.diag([0.010267495893, 0.010267495893, 0.00666, 3.7, 3.7, 3.7])
G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])
Glist = [G1, G2, G3, G4, G5, G6]

# Screw axes for each joint (Slist)
Slist = np.array([
    [0,         0,         0,         0,        0,        0],
    [0,         1,         1,         1,        0,        1],
    [1,         0,         0,         0,       -1,        0],
    [0, -0.089159, -0.089159, -0.089159, -0.10915, 0.005491],
    [0,         0,         0,         0,  0.81725,        0],
    [0,         0,     0.425,   0.81725,        0,  0.81725]
])

# Gravity vector (pointing down along -z)
g = np.array([0, 0, -9.81])
# Zero joint torques
tau = np.zeros(6)
# Zero external force at the end-effector
Ftip = np.zeros(6)

def simulate_robot(q_initial, total_time, dt, filename):
    """
    Simulates robot dynamics starting from q_initial with zero initial velocities,
    and writes the joint angles to a CSV file.
    Uses the Störmer–Verlet integration method.
    
    Args:
      q_initial: list of initial joint positions (6 values)
      total_time: simulation duration (seconds)
      dt: integration time step (seconds)
      filename: name of CSV file to save data
    """
    steps = int(total_time / dt)
    n_joints = len(q_initial)
    q = np.array(q_initial, dtype=float)
    dq = np.zeros(n_joints)
    trajectory = [q.copy()]
    
    # Compute initial acceleration
    ddq = mr.ForwardDynamics(q, dq, tau, g, Ftip, Mlist, Glist, Slist)
    
    for i in range(steps):
        # 1. Update velocity to half step
        dq_half = dq + 0.5 * ddq * dt
        # 2. Update position
        q_new = q + dq_half * dt
        # 3. Compute new acceleration at new position
        ddq_new = mr.ForwardDynamics(q_new, dq_half, tau, g, Ftip, Mlist, Glist, Slist)
        # 4. Complete velocity update
        dq_new = dq_half + 0.5 * ddq_new * dt
        
        # Update states for next step
        q, dq, ddq = q_new, dq_new, ddq_new
        
        trajectory.append(q.copy())
    
    # Save trajectory to CSV file
    np.savetxt(filename, trajectory, delimiter=",", fmt="%.6f")
    print(f"Simulation data saved to file: {filename}")

# Simulation 1: Falling from home (zero) configuration for 3 seconds
simulate_robot(q_initial=[0, 0, 0, 0, 0, 0], total_time=3.0, dt=0.001, filename="simulation1.csv")

# Simulation 2: Falling from configuration with only second joint at −1 radian for 5 seconds
simulate_robot(q_initial=[0, -1.0, 0, 0, 0, 0], total_time=5.0, dt=0.001, filename="simulation2.csv")
