from robot import Robot_Sim
from grid import Grid
from utils import rotate_point, grid_distance, find_centroid
import math
import random


class PidController:
    """
    PID controller class for controlling the robot's linear and angular velocities.
    """

    def __init__(
        self, linear_kp, linear_ki, linear_kd, angular_kp, angular_ki, angular_kd
    ):
        self.linear_error = 0
        self.previous_linear_error = 0
        self.integral_linear_error = 0

        self.angular_error = 0
        self.integral_angular_error = 0
        self.previous_angular_error = 0

        self.stopped = False

        self.linear_kp = linear_kp
        self.linear_ki = linear_ki
        self.linear_kd = linear_kd
        self.angular_kp = angular_kp
        self.angular_ki = angular_ki
        self.angular_kd = angular_kd

    def compute_angular_error(self, current_pose, next_waypoint):
        x_robot, y_robot, theta_robot = current_pose
        dx_world = next_waypoint[0] - x_robot
        dy_world = next_waypoint[1] - y_robot
        dx_robot, dy_robot = rotate_point(dx_world, dy_world, theta_robot)
        return math.atan2(dy_robot, dx_robot)

    def compute_linear_error(self, current_pose, goal_point):
        x_robot, y_robot, _ = current_pose
        dx_world = goal_point[0] - x_robot
        dy_world = goal_point[1] - y_robot
        return math.sqrt(dx_world**2 + dy_world**2)

    def linear_controller(self, pose, goal_point):
        dist_to_goal = self.compute_linear_error(pose, goal_point)
        self.integral_linear_error += dist_to_goal
        derivative_error = dist_to_goal - self.previous_linear_error
        v = (
            self.linear_kp * dist_to_goal
            + self.linear_ki * self.integral_linear_error
            + self.linear_kd * derivative_error
        )
        self.previous_linear_error = dist_to_goal
        return v

    def angular_controller(self, pose, waypoint):
        angular_error = self.compute_angular_error(pose, waypoint)
        self.integral_angular_error += angular_error
        derivative_error = angular_error - self.previous_angular_error
        omega = (
            self.angular_kp * angular_error
            + self.angular_ki * self.integral_angular_error
            + self.angular_kd * derivative_error
        )
        self.previous_angular_error = angular_error
        return omega


def get_neighbors(cell, grid, coordinates):
    """
    Get neighbors of a given cell
    """
    x, y = cell
    neighbors = [(x - 1, y - 1), (x, y - 1), (x - 1, y), (x + 1, y)]
    return [
        neighbor
        for neighbor in neighbors
        if grid.is_in(neighbor[0], neighbor[1]) and neighbor in coordinates
    ]


def separate_frontiers(coordinates, grid):
    label_grid = {}
    label_equivalences = {}
    next_label = 1

    def find_root(label):
        """
        Find the root label for equivalence resolution with path compression.
        """
        if label != label_equivalences.get(label, label):
            label_equivalences[label] = find_root(label_equivalences[label])
        return label_equivalences.get(label, label)

    def union_labels(label1, label2):
        """
        Merge two labels into a single equivalence class.
        """
        root1 = find_root(label1)
        root2 = find_root(label2)
        if root1 != root2:
            label_equivalences[root2] = root1

    for cell in sorted(coordinates, key=lambda c: (c[1], c[0])):
        x, y = cell
        neighbors = get_neighbors(cell, grid, coordinates)

        neighbor_labels = [
            label_grid[neighbor] for neighbor in neighbors if neighbor in label_grid
        ]

        if not neighbor_labels:
            label_grid[cell] = next_label
            next_label += 1
        elif len(neighbor_labels) == 1:
            label_grid[cell] = neighbor_labels[0]
        else:
            min_label = min(neighbor_labels)
            label_grid[cell] = min_label
            for label in neighbor_labels:
                if label != min_label:
                    union_labels(min_label, label)

    resolved_labels = {label: find_root(label) for label in set(label_grid.values())}

    for cell in label_grid:
        label_grid[cell] = resolved_labels[label_grid[cell]]

    components = {}
    for cell, label in label_grid.items():
        if label not in components:
            components[label] = []
        components[label].append(cell)

    return list(components.values())


def get_wheel_velocities(robbie, coord):
    dx_world = coord[0] - robbie.x
    dy_world = coord[1] - robbie.y
    dx_robot, dy_robot = rotate_point(dx_world, dy_world, robbie.h)
    dist_to_coord = math.sqrt(dx_robot**2 + dy_robot**2)
    angle = math.atan2(dy_robot, dx_robot)

    threshold = 0.1
    linear_v = robbie.pid_controller.linear_controller(
        (robbie.x, robbie.y, robbie.h), coord
    )
    if abs(angle) > threshold:
        linear_v = 0
    w = robbie.pid_controller.angular_controller((robbie.x, robbie.y, robbie.h), coord)

    vl = linear_v - robbie.wheel_dist / 2 * w
    vr = linear_v + robbie.wheel_dist / 2 * w
    return vr, vl


def frontier_planning(robbie: Robot_Sim, grid: Grid):
    frontier_cells = [
        cell
        for cell in robbie.get_free_cells_in_fov(grid)
        if any(
            not grid.is_in(*neighbor) or not grid.is_free(*neighbor)
            for neighbor in get_neighbors(cell)
        )
    ]
    components = separate_frontiers(frontier_cells, grid)
    centroids = [find_centroid(component) for component in components]
    centroids = sorted(
        centroids, key=lambda c: grid_distance(robbie.x, robbie.y, c[0], c[1])
    )
    robbie.next_coord = next(
        (c for c in centroids if not grid.is_occupied(int(c[0]), int(c[1]))),
        random.choice(frontier_cells),
    )
    grid.centroid = robbie.next_coord
    return robbie, robbie.next_coord


def exploration_state_machine(robbie, grid):
    if not hasattr(robbie, "pid_controller"):
        robbie.pid_controller = PidController(0.5, 0.01, 0.1, 1.0, 0.01, 0.1)

    free_cells = robbie.get_free_cells_in_fov(grid)

    if not robbie.next_coord or (robbie.x, robbie.y) == robbie.next_coord:
        robbie, _ = frontier_planning(robbie, grid)

    if grid.is_collision_with_obstacles((robbie.x, robbie.y), robbie.next_coord):
        path = grid.rrt((robbie.x, robbie.y), robbie.next_coord)
        if path:
            robbie.next_coord = path[1] if len(path) > 1 else path[0]

    vr, vl = get_wheel_velocities(robbie, robbie.next_coord)
    robbie.move_diff_drive(grid, vl, vr, robbie.TIMESTEP)
    return robbie
