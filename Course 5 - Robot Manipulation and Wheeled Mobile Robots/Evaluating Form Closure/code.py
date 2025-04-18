import numpy as np
import matplotlib.pyplot as plt

def form_closure(contact_points, normals):
    """
    Check form closure based on contact points and normals.
    """
    num_contacts = len(contact_points)
    
    # Construct the matrix for analysis
    A = np.zeros((2 * num_contacts, 3))  # 2 contacts, 3 degrees of freedom (X, Y, rotation)
    
    log_text = "Matrix for form closure check:\n"
    for i in range(num_contacts):
        contact = np.append(contact_points[i], 0)  # Add a zero Z component
        normal = np.append(normals[i], 0)  # Add a zero Z component
        A[2*i, 0] = normal[0]
        A[2*i, 1] = normal[1]
        
        # Cross product to calculate rotational force
        cross_prod = np.cross(contact, normal)  
        A[2*i + 1, 2] = cross_prod[2]  # Save only the Z component
        
        log_text += f"Contact {i+1}: {contact} - Normal: {normal}\n"
        log_text += f"Cross product for contact {i+1}: {cross_prod}\n"

    log_text += "\nMatrix A:\n"
    log_text += str(A)
    
    # Check the rank of the matrix (if the rank is less than 3, movement is possible)
    rank = np.linalg.matrix_rank(A)
    log_text += f"\nRank of matrix A: {rank}\n"
    
    # Check geometric integrity (angle between normals)
    if not check_geometry(contact_points, normals):
        return False, log_text
    
    return rank == 3, log_text

def check_geometry(contact_points, normals):
    """
    Check the geometric integrity: the angle between normals should not be too small.
    """
    for i in range(len(contact_points)):
        for j in range(i + 1, len(contact_points)):
            dot_product = np.dot(normals[i], normals[j])
            angle = np.arccos(dot_product) * (180.0 / np.pi)
            if angle < 10:  # If the angle between the normals is too small
                return False
    return True

def visualize(contact_points, normals, result, test_number):
    fig, ax = plt.subplots()
    for i, contact in enumerate(contact_points):
        ax.plot(contact[0], contact[1], 'ro')  # Contact point
        ax.quiver(contact[0], contact[1], normals[i][0], normals[i][1], angles='xy', scale_units='xy', scale=1)  # Normal

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Form Closure: {result}")
    plt.grid(True)
    
    # Save the image
    plt.savefig(f"form_closure_test_{test_number}.png")
    plt.close()  # Close the figure after saving

def run_test(test_number, contact_points, normals):
    # Run the form closure check
    result, log_text = form_closure(contact_points, normals)
    
    # Write logs and result to a file
    with open(f"log_test_{test_number}.txt", "w") as log_file:
        log_file.write(log_text)
        if result:
            log_file.write(f"\nTest {test_number}: Object is in form closure.\n")
        else:
            log_file.write(f"\nTest {test_number}: Object is not in form closure.\n")
    
    # Visualize and save the result
    visualize(contact_points, normals, "True" if result else "False", test_number)

# Test cases
def test_case_1():
    contact_points = [np.array([1, 1]), np.array([-1, 1])]
    normals = [np.array([1, 0]), np.array([0, 1])]
    run_test(1, contact_points, normals)

def test_case_2():
    contact_points = [np.array([1, 1]), np.array([-1, 0])]  # Shifted contact point
    normals = [np.array([1, 0]), np.array([0, 1])]
    run_test(2, contact_points, normals)

def test_case_3():
    contact_points = [np.array([2, 3]), np.array([-2, 3]), np.array([1, -1])]
    normals = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    run_test(3, contact_points, normals)

# Run all tests
test_case_1()
test_case_2()
test_case_3()
