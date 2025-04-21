from collections import namedtuple
from typing import List
import numpy as np
from scipy.optimize import linprog

StaticMass = namedtuple("StaticMass", ["m", "x_pos", "y_pos"])
Contact = namedtuple("Contact", ["body_a", "body_b", "x_contact", "y_contact", "normal_angle", "friction_coefficient"])

def find_contacts(contacts: List[Contact], body: int):
    """Find all contacts of a given body"""
    return [c for c in contacts if c.body_a == body or c.body_b == body]


def is_body_stable(m, contacts):
    """Check stability of given body masses and contacts."""
    g = 9.81
    num_bodies = len(m)
    for i in range(num_bodies):
        contact_bodies = find_contacts(contacts, i + 1)
        num_k = len(contact_bodies) * 2

        f = np.ones(num_k)
        A = -np.identity(num_k)
        b = -np.ones(num_k)
        F = []

        for contact_b in contact_bodies:
            x = contact_b.x_contact
            y = contact_b.y_contact
            ang = contact_b.normal_angle
            if i + 1 == contact_b.body_b:
                ang -= np.pi

            friction_coeff = contact_b.friction_coefficient
            theta = np.arctan2(friction_coeff, 1)

            f1 = [np.sin(ang + theta) * x - np.cos(ang + theta) * y,
                  np.cos(ang + theta),
                  np.sin(ang + theta)]
            f2 = [np.sin(ang - theta) * x - np.cos(ang - theta) * y,
                  np.cos(ang - theta),
                  np.sin(ang - theta)]

            F.extend([f1, f2])

        F = np.array([np.array(xi) for xi in F]).T

        Aeq = F
        static_mass = m[i]
        beq = [static_mass.m * static_mass.x_pos * g, 0, static_mass.m * g]
        k = linprog(f, A, b, Aeq, beq, method='highs-ipm')
        return k


open("output.txt", "w").close()

def log_stability_test(m, contacts):
    """Logging result of stability test"""
    if not hasattr(log_stability_test, "test_counter"):
        log_stability_test.test_counter = 1

    with open("output.txt", "a") as f:
        f.write(f"Test {log_stability_test.test_counter}\n")
        f.write("Masses data:\n")
        for ms in m:
            f.write(f"{ms}\n")
        f.write("Contacts data:\n")
        for c in contacts:
            f.write(f"{c}\n")

        k = is_body_stable(m, contacts)
        result = "Result:\nThe assembly "
        result += "remains standing\n" if k.success else "would collapse\n"
        f.write(result + "\n")

    log_stability_test.test_counter += 1


if __name__ == "__main__":
    # Test examples from the book
    m_0 = [StaticMass(2, 25, 35),
           StaticMass(5, 66, 42)]
    contacts_0 = [Contact(1, 0, 0, 0, np.pi / 2, 0.1),
                  Contact(1, 2, 60, 60, np.pi, 0.5),
                  Contact(2, 0, 72, 0, np.pi / 2, 0.5),
                  Contact(2, 0, 60, 0, np.pi / 2, 0.5)]
    log_stability_test(m_0, contacts_0) # expected collapses state

    m_1 = [StaticMass(2, 25, 35),
           StaticMass(10, 66, 42)]
    contacts_1 = [Contact(1, 0, 0, 0, np.pi / 2, 0.5),
                  Contact(1, 2, 60, 60, np.pi, 0.5),
                  Contact(2, 0, 72, 0, np.pi / 2, 0.5),
                  Contact(2, 0, 60, 0, np.pi / 2, 0.5)]
    log_stability_test(m_1, contacts_1) # expected stable state
