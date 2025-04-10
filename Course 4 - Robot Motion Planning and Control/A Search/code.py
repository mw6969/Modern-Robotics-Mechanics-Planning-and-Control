import csv
import math
import heapq
import sys
import os

# Reads nodes from the file nodes.csv.
# Lines starting with '#' are considered comments and are ignored.
# Returns a dictionary: node_id -> (x, y, heuristic)
def read_nodes(filename):
    nodes = {}
    try:
        with open(filename, newline='') as csvfile:
            # Отфильтровываем комментарии
            lines = [line for line in csvfile if not line.strip().startswith("#")]
            reader = csv.reader(lines)
            for row in reader:
                if len(row) != 4:
                    raise ValueError(f"Invalid number of columns in nodes.csv: {row}")
                node_id = int(row[0])
                x = float(row[1])
                y = float(row[2])
                heuristic = float(row[3])
                nodes[node_id] = (x, y, heuristic)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        sys.exit(1)
    return nodes

# Reads edges from the file edges.csv.
# Lines starting with '#' are considered comments and are ignored.
# Returns a list of tuples: (id1, id2, cost)
def read_edges(filename):
    edges = []
    try:
        with open(filename, newline='') as csvfile:
            # Отфильтровываем комментарии
            lines = [line for line in csvfile if not line.strip().startswith("#")]
            reader = csv.reader(lines)
            for row in reader:
                if len(row) != 3:
                    raise ValueError(f"Invalid number of columns in edges.csv: {row}")
                id1 = int(row[0])
                id2 = int(row[1])
                cost = float(row[2])
                edges.append((id1, id2, cost))
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        sys.exit(1)
    return edges

# Reads obstacles from obstacles.csv.
# Lines starting with '#' are considered comments and are ignored.
# Returns a list of tuples: (center_x, center_y, radius) where radius = diameter/2
def read_obstacles(filename):
    obstacles = []
    try:
        with open(filename, newline='') as csvfile:
            # Отфильтровываем комментарии
            lines = [line for line in csvfile if not line.strip().startswith("#")]
            reader = csv.reader(lines)
            for row in reader:
                if len(row) != 3:
                    raise ValueError(f"Invalid number of columns in obstacles.csv: {row}")
                cx = float(row[0])
                cy = float(row[1])
                diameter = float(row[2])
                radius = diameter / 2.0
                obstacles.append((cx, cy, radius))
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        sys.exit(1)
    return obstacles

# Checks if a point (x, y) is inside any obstacle.
def point_in_obstacle(x, y, obstacles):
    for cx, cy, radius in obstacles:
        if math.hypot(x - cx, y - cy) < radius:
            return True
    return False

# Checks if the line segment between p1 and p2 intersects a circle (center, radius).
def segment_intersects_circle(p1, p2, center, radius):
    (x1, y1) = p1
    (x2, y2) = p2
    (cx, cy) = center

    # Vector from p1 to p2
    dx = x2 - x1
    dy = y2 - y1

    # If the segment is very short, use the distance from p1 to the center
    if dx == 0 and dy == 0:
        return math.hypot(x1 - cx, y1 - cy) < radius

    # Calculate the parameter t for the projection of the center onto the line
    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    # Clamp t to the segment [0, 1]
    t = max(0, min(1, t))
    # Find the nearest point on the segment to the circle center
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    # Calculate the distance from the nearest point to the center
    distance = math.hypot(nearest_x - cx, nearest_y - cy)
    return distance < radius

# Checks if an edge (between two nodes) collides with any obstacle.
def edge_collides(p1, p2, obstacles):
    for cx, cy, radius in obstacles:
        # If one of the endpoints is already inside the obstacle, the edge is considered colliding.
        if point_in_obstacle(p1[0], p1[1], [(cx, cy, radius)]) or point_in_obstacle(p2[0], p2[1], [(cx, cy, radius)]):
            return True
        if segment_intersects_circle(p1, p2, (cx, cy), radius):
            return True
    return False

# Builds the graph considering filtered nodes and obstacle collisions on edges.
# Graph is represented as a dictionary: node_id -> list of tuples (neighbor_id, cost)
def build_graph(nodes, edges, obstacles):
    graph = {node_id: [] for node_id in nodes.keys()}
    for id1, id2, cost in edges:
        # Consider the edge only if both nodes are present (not removed due to obstacles)
        if id1 in nodes and id2 in nodes:
            p1 = (nodes[id1][0], nodes[id1][1])
            p2 = (nodes[id2][0], nodes[id2][1])
            # Check if the edge collides with any obstacle
            if not edge_collides(p1, p2, obstacles):
                # Since the graph is undirected, add the edge in both directions.
                graph[id1].append((id2, cost))
                graph[id2].append((id1, cost))
    return graph

# Implementation of the A* algorithm.
def astar(graph, nodes, start, goal):
    # Check that start and goal are in the graph.
    if start not in graph or goal not in graph:
        return None

    # g_score: the cost of the shortest path found so far from start to the node.
    g_score = {node: math.inf for node in graph}
    g_score[start] = 0

    # came_from: stores the parent of each node to reconstruct the path.
    came_from = {}

    # Priority queue (min-heap) for the open set.
    # Each element is a tuple: (f, g, node)
    open_list = []
    # f = g + heuristic (heuristic is taken from nodes)
    heuristic_start = nodes[start][2]
    heapq.heappush(open_list, (heuristic_start, 0, start))
    
    # Closed set to track nodes already evaluated.
    closed = set()

    while open_list:
        current_f, current_g, current = heapq.heappop(open_list)

        if current == goal:
            # Reconstruct the path from goal to start.
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))
        
        closed.add(current)

        for neighbor, edge_cost in graph[current]:
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # Calculate f = g + h. The heuristic is taken from the nodes data.
                heuristic = nodes[neighbor][2]
                f = tentative_g + heuristic
                heapq.heappush(open_list, (f, tentative_g, neighbor))
    return None

def main():
    # File names are defined.
    nodes_file = "nodes.csv"
    edges_file = "edges.csv"
    obstacles_file = "obstacles.csv"
    output_file = "path.csv"

    # Read input data.
    nodes = read_nodes(nodes_file)
    edges = read_edges(edges_file)
    obstacles = read_obstacles(obstacles_file)

    # Filter nodes: exclude those that are inside any obstacle.
    valid_nodes = {}
    for node_id, (x, y, heuristic) in nodes.items():
        if not point_in_obstacle(x, y, obstacles):
            valid_nodes[node_id] = (x, y, heuristic)
        else:
            print(f"Node {node_id} is excluded because it is inside an obstacle.")
    
    # According to the assignment, the start node is 1 and the goal node is the one with the highest ID.
    start = 1
    goal = max(nodes.keys())
    if start not in valid_nodes:
        print("The start node (1) is inside an obstacle or is missing.")
    if goal not in valid_nodes:
        print(f"The goal node ({goal}) is inside an obstacle or is missing.")

    if start not in valid_nodes or goal not in valid_nodes:
        # If either start or goal is missing, there is no valid path.
        with open(output_file, "w") as f:
            f.write("1")
        sys.exit(0)

    # Use valid nodes for further computations.
    nodes = valid_nodes

    # Build the graph considering obstacles for the edges.
    graph = build_graph(nodes, edges, obstacles)

    # Run the A* search.
    path = astar(graph, nodes, start, goal)

    # Write the output file.
    try:
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if path is None:
                # If no path is found, write only the start node.
                writer.writerow([start])
            else:
                writer.writerow(path)
    except Exception as e:
        print(f"Error writing file {output_file}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
