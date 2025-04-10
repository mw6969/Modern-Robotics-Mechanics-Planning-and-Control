def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 4
    Vb = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    thetalists_matrix = np.empty((maxiterations, thetalist.size))
    while err and i < maxiterations:
        thetalist = thetalist + np.dot(np.linalg.pinv(JacobianBody(Blist, thetalist)), Vb)
        thetalists_matrix[i, :] = thetalist
        end_effector_config = FKinBody(M, Blist, thetalist)
        Vb = se3ToVec(MatrixLog6(np.dot(TransInv(end_effector_config), T)))
        omega_b = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
        v_b = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
        err = omega_b > eomg or v_b > ev
        
        print(f"Iteration {i}:")
        print(f"joint vector: {', '.join(f'{theta:.3f}' for theta in thetalist)}")
        print(f"SE(3) end−effector config: [" + "".join(f"[{' '.join(f'{x:.3f}' for x in row)}]" for row in end_effector_config) + "]")
        print(f"error twist V_b: {', '.join(f'{item:.3f}' for item in Vb)}")
        print(f"angular error magnitude ||omega_b||: {omega_b:.3f}")
        print(f"linear error magnitude ||v_b||: {v_b:.3f}\n")       
        
        i = i + 1
    np.savetxt("iterates.csv", thetalists_matrix, delimiter=",", fmt="%.3f")
    return (thetalist, not err)