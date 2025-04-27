import os
import sys
import argparse
import configparser
import numpy as np
import matplotlib.pyplot as plt
import modern_robotics as mr


# --- Matrix rotation ---
def rotate_matrix(R, phi, axis):
    if phi==0:
        return R
    c, s = np.cos(phi), np.sin(phi)
    if axis=='x':
        rot = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    elif axis=='y':
        rot = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    else:
        rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return rot.dot(R)


# --- Segment duration ---
def segment_duration(X_start, X_end):
    R1,p1 = mr.TransToRp(X_start)
    R2,p2 = mr.TransToRp(X_end)
    d  = np.linalg.norm(np.array(p2)-np.array(p1))
    theta = np.arccos((np.trace(R2)-1)/2)-np.arccos((np.trace(R1)-1)/2)
    t_lin = d/0.25
    t_ang = theta/0.25
    return max(1, int(max(t_lin,t_ang)))


# --- Trajectory segment ---
def trajectory_segment(X_start, X_end, tf, k, method, degree, gripper_state):
    N = int(tf*k/0.01)
    if X_end is None:
        traj = [X_start.copy() for _ in range(N)]
    else:
        if method=='screw':
            traj = mr.ScrewTrajectory(X_start, X_end, tf, N, degree)
        else:
            traj = mr.CartesianTrajectory(X_start, X_end, tf, N, degree)
    return traj, [gripper_state]*len(traj)


# --- Trajectory generation ---
def generate_trajectory(tse_init, tsc_init, tsc_final, method, k, degree, pickup_angle, z_standoff, shift_x, shift_z):
    Rse_init, pse_init = mr.TransToRp(tse_init)
    Rc_init, pc_init = mr.TransToRp(tsc_init)
    Rc_final, pc_final = mr.TransToRp(tsc_final)

    # 1: approach standoff
    R1 = rotate_matrix(Rse_init, pickup_angle, 'y')
    p1 = [pc_init[0], pc_init[1], pc_init[2]+z_standoff]
    X1 = mr.RpToTrans(R1, p1)
    tf1 = segment_duration(tse_init, X1)
    seg1, grip1 = trajectory_segment(tse_init, X1, tf1, k, method, degree, 0)

    # 2: down to cube
    X2 = mr.RpToTrans(R1, [pc_init[0]+shift_x, pc_init[1], pc_init[2]+shift_z])
    tf2 = segment_duration(X1, X2)
    seg2, grip2 = trajectory_segment(X1, X2, tf2, k, method, degree, 0)

    # 3: grab
    seg3, grip3 = trajectory_segment(X2, None, 2, k, method, degree, 1)

    # 4: lift up
    tf4 = segment_duration(X2, X1)
    seg4, grip4 = trajectory_segment(X2, X1, tf4, k, method, degree, 1)

    # 5: to position above final
    R5 = rotate_matrix(R1, -np.pi/2, 'z')
    p5 = [pc_final[0], pc_final[1], pc_final[2]+z_standoff]
    X5 = mr.RpToTrans(R5, p5)
    tf5 = segment_duration(X1, X5)
    seg5, grip5 = trajectory_segment(X1, X5, tf5, k, method, degree, 1)

    # 6: lower the cube
    X6 = mr.RpToTrans(R5, [pc_final[0], pc_final[1]-shift_x, pc_final[2]-shift_z])
    tf6 = segment_duration(X5, X6)+1
    seg6, grip6 = trajectory_segment(X5, X6, tf6, k, method, degree, 1)

    # 7: let go
    seg7, grip7 = trajectory_segment(X6, None, 2, k, method, degree, 0)

    # 8: step back
    tf8 = segment_duration(X6, X5)
    seg8, grip8 = trajectory_segment(X6, X5, tf8, k, method, degree, 0)

    trajectory = seg1+seg2+seg3+seg4+seg5+seg6+seg7+seg8
    grippers  = grip1+grip2+grip3+grip4+grip5+grip6+grip7+grip8

    # return a list of Tse matrices
    return trajectory, grippers


# --- Decoding and limiting ---
def decode_configuration(q):
    return np.array(q[0:3]), np.array(q[3:8]), np.array(q[8:12])


def decode_control(u):
    return np.array(u[0:4]), np.array(u[4:9])


def vel_limits(qw_dot, qa_dot, max_w_dot, max_a_dot):
    return (np.clip(qw_dot, -max_w_dot, max_w_dot), np.clip(qa_dot, -max_a_dot, max_a_dot))


# --- Kinematics and Jacobian ---
def forward_kinematics(qc, qa, m_0e, b, t_b0):
    phi, x, y = qc
    T_sb = np.array([[np.cos(phi), -np.sin(phi), 0, x],[np.sin(phi), np.cos(phi), 0, y],[0, 0, 1, 0.0963],[0, 0, 0, 1]])
    T_0e = mr.FKinBody(m_0e, b, qa)
    return T_sb.dot(t_b0).dot(T_0e)


def compute_je(qa, b, f6, m_0e, t_b0):
    T_0e = mr.FKinBody(m_0e, b, qa)
    J_base = mr.Adjoint(np.linalg.inv(T_0e).dot(np.linalg.inv(t_b0))).dot(f6)
    J_arm = mr.JacobianBody(b, qa)
    return np.hstack((J_base, J_arm))


# --- Odometry and Status Update ---
def odometry(qc, qw_delta, f):
    phi, x, y = qc
    twist = f.dot(qw_delta)
    wbz, vbx, vby = twist
    if abs(wbz) < 1e-9:
        qc_delta = np.array([0., vbx, vby])
    else:
        qc_delta = np.array([wbz,(vbx*np.sin(wbz) + vby*(np.cos(wbz)-1))/wbz,(vby*np.sin(wbz) + vbx*(1-np.cos(wbz)))/wbz])
    R_sb = np.array([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
    return qc + R_sb.dot(qc_delta)


def next_state(q, u, dt, f, max_w, max_a):
    qc, qa, qw = decode_configuration(q)
    qw_dot, qa_dot = decode_control(u)
    qw_dot, qa_dot = vel_limits(qw_dot, qa_dot, max_w, max_a)
    qc_new = odometry(qc, qw_dot*dt, f)
    qa_new = qa + qa_dot*dt
    qw_new = qw + qw_dot*dt
    return np.concatenate([qc_new, qa_new, qw_new])


# --- Motion control ---
def feedforward_pi_control(x, xd, xd_next, kp, ki, dt, xe_integral):
    vd = mr.se3ToVec((1/dt) * mr.MatrixLog6(np.dot(mr.TransInv(xd), xd_next)))
    x_err = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(x), xd)))
    feedforward = np.dot(mr.Adjoint(np.dot(mr.TransInv(x), xd)), vd)
    v = feedforward + kp @ x_err + ki @ (xe_integral * dt)
    xe_integral += x_err
    return v, x_err, xe_integral


def compute_control(q, qc, qa, xd, xd_next, params, xe_integral):
    x = forward_kinematics(qc, qa, params['m_0e'], params['b'], params['t_b0'])
    v, x_err, xe_integral = feedforward_pi_control(
        x, xd, xd_next, params['kp'], params['ki'], params['dt'], xe_integral
    )
    j_e = compute_je(qa, params['b'], params['f6'], params['m_0e'], params['t_b0'])
    u = np.linalg.pinv(j_e, 0.0001) @ v
    return u, x_err, xe_integral


# --- Simulation of motion along a trajectory ---
def simulate_trajectory(q_init, planned_traj, grippers, params):
    q = np.array(q_init)
    xe_int = np.zeros(6)
    traj_d, err_d, u_d = [], [], []
    for i in range(len(planned_traj)-1):
        Xd, Xd_next = planned_traj[i], planned_traj[i+1]
        qc, qa, _ = decode_configuration(q)
        u, err, xe_int = compute_control(q, qc, qa, Xd, Xd_next, params, xe_int)
        q = next_state(q, u, params['dt'], params['f'], params['max_w_dot'], params['max_a_dot'])
        # вместо [0] теперь берем реальный gripper-флаг
        g = grippers[i]
        traj_d.append(np.concatenate([q, [g]]))
        err_d.append(err)
        u_d.append(u)
    return np.array(traj_d), np.array(err_d), np.array(u_d)


# --- Auxiliary functions ---
def create_dirs(base_dir, suffix):
    results_dir = os.path.join(base_dir, suffix)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def load_config(config_path):
    config = configparser.ConfigParser()
    if not os.path.isfile(config_path):
        print(f"Config '{config_path}' not found.")
        sys.exit(1)
    config.read(config_path)

    def parse_value(value):
        try:
            return eval(value, {"np": np, "array": np.array, "__builtins__": {}})
        except:
            try:
                return float(value)
            except ValueError:
                return value.strip("'\"")

    def get_param(section, option):
        return parse_value(config[section][option])

    return get_param


def init_robot_and_controller(params):
    robot_params = {
        'd_l': params('robot', 'd_l'),
        'd_w': params('robot', 'd_w'),
        'r': params('robot', 'r'),
        'm_0e': params('robot', 'm_0e'),
        'b': params('robot', 'b'),
        't_b0': params('robot', 't_b0'),
        'joint_limits': params('robot', 'joint_limits')
    }
    control_params = {
        'dt': params('control', 'dt'),
        'k': params('control', 'k'),
        'method': params('control', 'method'),
        'degree': params('control', 'degree'),
        'z_standoff': params('control', 'z_standoff'),
        'pickup_angle': params('control', 'pickup_angle'),
        'max_w_dot': params('control', 'max_w_dot'),
        'max_a_dot': params('control', 'max_a_dot'),
        'control_type': params('control', 'control_type'),
        'test_joint_limits_flag': params('control', 'test_joint_limits_flag'),
        'kp': params('control', 'kp'),
        'ki': params('control', 'ki'),
        'kd': params('control', 'kd')
    }
    return robot_params, control_params


def save_results(results_dir, data_dict):
    for name, data in data_dict.items():
        path = os.path.join(results_dir, f"{name}.csv")
        np.savetxt(path, data, delimiter=',')


def plot_data(data_path, save_path, ylabel, labels=None):
    data = np.loadtxt(data_path, delimiter=',')
    steps = np.arange(data.shape[0])
    plt.figure()
    if labels:
        for i, lab in enumerate(labels): plt.plot(steps, data[:,i], label=lab)
        plt.legend()
    else:
        for i in range(data.shape[1]): plt.plot(steps, data[:,i])
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--mode",required=True,help="mode: best, overshoot, newTask")
    args = parser.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    rd = create_dirs(os.path.join(base,'results'),args.mode)
    params = load_config(os.path.join(base,f"{args.mode}.ini"))
    rp, cp = init_robot_and_controller(params)
    
    # build f and f6 
    d_l, d_w, r = rp['d_l'], rp['d_w'], rp['r']
    f = (r/4)*np.array([[-1/(d_l+d_w),1/(d_l+d_w),1/(d_l+d_w),-1/(d_l+d_w)],[1,1,1,1],[-1,1,-1,1]])
    f6 = np.zeros((6,4)); f6[2:5,:]=f
    cp.update(rp); cp['f'],cp['f6']=f,f6

    # initial conditions
    q_init = params('control','q')
    tse_i, tsc_i, tsc_f = params('control','tse_init'),params('control','tsc_init'),params('control','tsc_final')
    
    # trajectory generation
    traj, grippers = generate_trajectory(
        tse_i, tsc_i, tsc_f,
        cp['method'], cp['k'], cp['degree'],
        cp['pickup_angle'], cp['z_standoff'],
        params('control','shift_x'), params('control','shift_z')
    )
    
    # simulation
    traj_d, err_d, u_d = simulate_trajectory(q_init, traj, grippers, cp)

    # save
    save_results(rd, {
        'trajectory': traj_d,
        'err': err_d,
        'controls': u_d,
        'desired_trajectory': np.array(traj).reshape((len(traj), -1))
    })

    # plot
    plot_data(os.path.join(rd,'err.csv'),   os.path.join(rd,'err.png'),   f"Error - {cp['control_type']}",[f'j{i+1}' for i in range(5)])
    plot_data(os.path.join(rd,'controls.csv'),os.path.join(rd,'controls.png'),f"Controls - {cp['control_type']}")

    print(f"Done! Results saved by path: {rd}")