import random
import math
import csv

# Global parameters
START = (-0.5, -0.5)
GOAL  = (0.5, 0.5)
X_LIMITS = (-0.5, 0.5)
Y_LIMITS = (-0.5, 0.5)
MAX_ITERATIONS = 10000
GOAL_SAMPLE_RATE = 0.1  # 10% samples will be the goal configuration
STEP_SIZE = 0.05        # step size when expanding the tree
COLLISION_THRESHOLD = 1e-6  # Threshold to detect significant collisions
ROBOT_RADIUS = 0.02

def load_obstacles(filename):
    """
    Reads obstacles from the file obstacles.csv.
    Each line contains: x, y, diameter.
    Returns a list of obstacles as tuples (x, y, radius).
    """
    obstacles = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x = float(row[0].strip())
            y = float(row[1].strip())
            diameter = float(row[2].strip())
            obstacles.append((x, y, diameter / 2.0))
    return obstacles

def euclidean_distance(p1, p2):
    """Computes the Euclidean distance between points p1 and p2."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_significant_collision(nearest_point, circle_center, radius, threshold=COLLISION_THRESHOLD):
    """
    Checks if the distance between nearest_point and the circle is within the inflated radius.
    """
    inflated_radius = radius + ROBOT_RADIUS  # ← увеличенный радиус
    distance = euclidean_distance(nearest_point, circle_center)
    return distance <= inflated_radius + threshold

def collision_check(p1, p2, obstacles):
    for (cx, cy, radius) in obstacles:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            if is_significant_collision(p1, (cx, cy), radius):
                return False
            continue
        t = ((cx - p1[0]) * dx + (cy - p1[1]) * dy) / (dx * dx + dy * dy)
        t_clamped = max(0, min(1, t))
        nearest_x = p1[0] + t_clamped * dx
        nearest_y = p1[1] + t_clamped * dy
        if is_significant_collision((nearest_x, nearest_y), (cx, cy), radius):
            return False
    return True

def is_too_close_to_obstacle(point, obstacles):
    """
    Checks if a point is within (radius + robot_radius) of any obstacle.
    """
    for (cx, cy, radius) in obstacles:
        if euclidean_distance(point, (cx, cy)) <= radius + ROBOT_RADIUS:
            return True
    return False

def sample_configuration(obstacles):
    """
    Generates a random configuration that is not too close to obstacles.
    """
    for _ in range(100):  # Try 100 times before giving up
        if random.random() < GOAL_SAMPLE_RATE:
            return GOAL
        x = random.uniform(X_LIMITS[0], X_LIMITS[1])
        y = random.uniform(Y_LIMITS[0], Y_LIMITS[1])
        point = (x, y)
        if not is_too_close_to_obstacle(point, obstacles):
            return point
    return GOAL  # fallback if all samples fail

def nearest_node(tree, config):
    """
    Finds the nearest node in the tree to the given configuration.
    'tree' is a list of nodes, each node is a dictionary:
      {'id': int, 'pos': (x, y), 'parent': parent_id or None}.
    Returns the index of the found node and the node itself.
    """
    min_dist = float("inf")
    nearest = None
    nearest_index = -1
    for idx, node in enumerate(tree):
        d = euclidean_distance(node['pos'], config)
        if d < min_dist:
            min_dist = d
            nearest = node
            nearest_index = idx
    return nearest_index, nearest

def steer(from_node, to_config, step_size=STEP_SIZE):
    """
    Creates a new configuration by moving from from_node towards to_config by step_size.
    If the distance is less than step_size, returns to_config.
    """
    from_pos = from_node['pos']
    d = euclidean_distance(from_pos, to_config)
    if d < step_size:
        return to_config
    theta = math.atan2(to_config[1] - from_pos[1], to_config[0] - from_pos[0])
    new_x = from_pos[0] + step_size * math.cos(theta)
    new_y = from_pos[1] + step_size * math.sin(theta)
    return (new_x, new_y)

def reconstruct_path(tree, goal_index):
    """
    Reconstructs the path from the node at index goal_index back to the start using the 'parent' field.
    Returns a list of node IDs in order from start to goal.
    """
    path = []
    current = tree[goal_index]
    while current is not None:
        path.append(current['id'])
        if current['parent'] is None:
            break
        parent = None
        for node in tree:
            if node['id'] == current['parent']:
                parent = node
                break
        current = parent
    path.reverse()
    return path

def write_output(tree, edges, path, nodes_filename="nodes.csv", edges_filename="edges.csv", path_filename="path.csv"):
    """
    Writes the results to files:
      nodes.csv – nodes with heuristic (Euclidean distance to the goal),
      edges.csv – a list of edges [ID1, ID2, cost],
      path.csv – the sequence of node IDs of the solution path.
      If no path is found, writes a single value 1 to path.csv.
    """
    with open(nodes_filename, 'w', newline='') as f_nodes:
        writer = csv.writer(f_nodes)
        for node in tree:
            heuristic = euclidean_distance(node['pos'], GOAL)
            writer.writerow([node['id'], node['pos'][0], node['pos'][1], heuristic])
    with open(edges_filename, 'w', newline='') as f_edges:
        writer = csv.writer(f_edges)
        for edge in edges:
            writer.writerow(edge)
    with open(path_filename, 'w', newline='') as f_path:
        writer = csv.writer(f_path)
        if path:
            writer.writerow(path)
        else:
            writer.writerow([1])

def rrt_planner(obstacles):
    """
    The main function of the RRT algorithm.
    Returns three lists:
      tree – a list of nodes (dictionaries),
      edges – a list of edges [ID1, ID2, cost],
      solution_path – a list of node IDs from start to goal if a path is found, otherwise an empty list.
    """
    tree = []
    edges = []
    node_id = 1
    tree.append({'id': node_id, 'pos': START, 'parent': None})
    node_id += 1

    for _ in range(MAX_ITERATIONS):
        q_rand = sample_configuration(obstacles)
        nearest_index, nearest = nearest_node(tree, q_rand)
        q_new = steer(nearest, q_rand, STEP_SIZE)
        if collision_check(nearest['pos'], q_new, obstacles):
            new_node = {'id': node_id, 'pos': q_new, 'parent': nearest['id']}
            tree.append(new_node)
            cost = euclidean_distance(nearest['pos'], q_new)
            edges.append([nearest['id'], node_id, cost])
            node_id += 1
            if euclidean_distance(q_new, GOAL) < STEP_SIZE and collision_check(q_new, GOAL, obstacles):
                new_node_goal = {'id': node_id, 'pos': GOAL, 'parent': new_node['id']}
                tree.append(new_node_goal)
                cost_to_goal = euclidean_distance(q_new, GOAL)
                edges.append([new_node['id'], node_id, cost_to_goal])
                node_id += 1
                return tree, edges, reconstruct_path(tree, len(tree) - 1)
    return tree, edges, []

def main():
    obstacles = load_obstacles("obstacles.csv")
    tree, edges, solution_path = rrt_planner(obstacles)
    write_output(tree, edges, solution_path)

if __name__ == "__main__":
    main()
