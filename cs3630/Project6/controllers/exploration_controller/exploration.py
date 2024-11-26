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
        """
        Compute the angular error between robot's current heading and direction to waypoint.

        Parameters:
            current_pose: (x, y, theta) representing robot's current position and heading
            next_waypoint: (x, y) representing target point to move towards

        Returns:
            angular_error: Angle (in radians) robot needs to turn to face the waypoint
        """
        x_robot, y_robot, theta_robot = current_pose
        dx_world = next_waypoint[0] - x_robot
        dy_world = next_waypoint[1] - y_robot
        dx_robot, dy_robot = rotate_point(dx_world, dy_world, theta_robot)
        angular_error = math.atan2(dy_robot, dx_robot)

        return angular_error

    def compute_linear_error(self, current_pose, goal_point):
        """
        Compute the linear distance error between robot and goal point.

        Parameters:
            current_pose: (x, y, theta) representing robot's current position and heading
            goal_point: (x, y) representing target point to move towards

        Returns:
            dist_to_coord: Euclidean distance from robot to goal point
        """
        x_robot, y_robot, theta_robot = current_pose
        dx_world = goal_point[0] - x_robot
        dy_world = goal_point[1] - y_robot

        dx_robot, dy_robot = rotate_point(dx_world, dy_world, theta_robot)
        dist_to_coord = math.sqrt(dx_robot**2 + dy_robot**2)

        return dist_to_coord

    def linear_controller(self, pose, goal_point):
        """
        Returns the linear velocity based on the robot's current pose and goal_point.

        Parameters:
            pose (np.array): Current pose (x, y, theta)
            goal_point (np.array): Goal pose at the end of the trajectory (x, y)

        Returns: linear_velocity (float)
        """

        self.linear_error = self.compute_linear_error(pose, goal_point)

        self.integral_linear_error += self.linear_error

        derivative_linear_error = self.linear_error - self.previous_linear_error

        v = (
            self.linear_kp * self.linear_error
            + self.linear_ki * self.integral_linear_error
            + self.linear_kd * derivative_linear_error
        )

        self.previous_linear_error = self.linear_error

        return v

    def angular_controller(self, pose, waypoint):
        """
        Returns the angular velocity based on the robot's current pose and next waypoint.

        Parameters:
            pose (np.array): Current pose (x, y, theta)
            waypoint (np.array): Next waypoint pose to navigate to (x, y)

        Returns: angular_velocity (float)
        """

        self.angular_error = self.compute_angular_error(pose, waypoint)

        self.integral_angular_error += self.angular_error

        derivative_angular_error = self.angular_error - self.previous_angular_error

        omega = (
            self.angular_kp * self.angular_error
            + self.angular_ki * self.integral_angular_error
            + self.angular_kd * derivative_angular_error
        )

        self.previous_angular_error = self.angular_error

        return omega


def separate_frontiers(coordinates, grid):
    """
    Separates out a list of cells into a list of frontiers using one-pass connected component labeling
    (see pseudocode in Frontier-based Exploration lecture)

    Parameters:
    coordinates (list of tuples): A list of coordinates to be separated into frontiers
    grid (Grid): The grid object

    Returns:
    components (list of separated frontiers): A list of frontiers
    """

    label_map = {}
    current_label = 1
    equivalences = {}

    def find_root(label):

        while label != equivalences.get(label, label):
            label = equivalences[label]
        return label

    for x, y in sorted(coordinates):
        if not grid.is_free(x, y):
            continue

        neighbors = [
            (x - 1, y - 1),
            (x, y - 1),
            (x - 1, y),
        ]
        neighbor_labels = [
            label_map.get(neighbor) for neighbor in neighbors if neighbor in label_map
        ]

        if neighbor_labels:
            smallest_label = min(neighbor_labels)
            label_map[(x, y)] = smallest_label

            for label in neighbor_labels:
                root = find_root(label)
                equivalences[root] = smallest_label

        else:
            label_map[(x, y)] = current_label
            equivalences[current_label] = current_label
            current_label += 1

    resolved_labels = {label: find_root(label) for label in equivalences.keys()}
    for coord in label_map:
        label_map[coord] = resolved_labels[label_map[coord]]

    components = {}
    for coord, label in label_map.items():
        if label not in components:
            components[label] = []
        components[label].append(coord)

    return list(components.values())


def get_wheel_velocities(robbie, coord):
    """
    Helper function to determine the velocities of the robot's left and right wheels.
    Arguments:
        robbie: instance of the robot
        coord (tuple): coordinate to move to (x,y)

    Returns:
        vr, vl: velocities of the robot's left and right wheels
    """

    dx_world = coord[0] - robbie.x
    dy_world = coord[1] - robbie.y
    dx_robot, dy_robot = rotate_point(dx_world, dy_world, robbie.h)
    dist_to_coord = math.sqrt(dx_robot**2 + dy_robot**2)

    angle = math.atan2(dy_robot, dx_robot)

    threshold = 0.1

    linear_v = robbie.pid_controller.linear_controller(
        (robbie.x, robbie.y, robbie.h), coord
    )
    if angle < -threshold or angle > threshold:
        linear_v = 0
    w = robbie.pid_controller.angular_controller((robbie.x, robbie.y, robbie.h), coord)

    vl = linear_v - robbie.wheel_dist / 2 * w
    vr = linear_v + robbie.wheel_dist / 2 * w
    return vr, vl


def get_neighbors(cell):
    """
    Get neighbors of a given cell
    """
    return [
        (cell[0] + 1, cell[1]),
        (cell[0] - 1, cell[1]),
        (cell[0], cell[1] + 1),
        (cell[0], cell[1] - 1),
    ]


def frontier_planning(robbie: Robot_Sim, grid: Grid):
    """
    Function for defining frontier planning.

    Arguments:
        robbie: instance of the robot
        grid: instance of the grid

    Returns:
        robbie: 'updated' instance of the robot
        OPTIONAL: robbie.next_coord: new destination coordinate

    Notes:
        The lecture notes should provide you with an ample description of frontier planning.
        You will also find many of the functions declared in 'grid.py' and 'utils.py' useful.

    """

    frontier_cells = []
    for cell in robbie.explored_cells:
        if grid.is_free(*cell):
            neighbors = get_neighbors(cell)
            for neighbor in neighbors:

                if grid.is_in(*neighbor) and (neighbor not in robbie.explored_cells):
                    frontier_cells.append(neighbor)

    frontier_cells = list(set(frontier_cells))

    if not frontier_cells:
        return robbie, None

    frontiers = separate_frontiers(frontier_cells, grid)
    centroids = [find_centroid(frontier) for frontier in frontiers]

    valid_centroids = [
        c
        for c in centroids
        if grid.is_free(int(c[0]), int(c[1]))
        and grid.is_in(int(c[0]), int(c[1]))
        and c != (robbie.x, robbie.y)
        and grid_distance(c[0], c[1], robbie.x, robbie.y) > 1.0
    ]

    if not valid_centroids:
        random_cell = random.choice(frontier_cells) if frontier_cells else None
        return robbie, random_cell

    centroids_with_utilities = []
    alpha1, alpha2 = 1, 0.5
    for centroid, frontier in zip(centroids, frontiers):
        distance = grid_distance(robbie.x, robbie.y, *centroid)
        utility = alpha1 * distance + alpha2 * len(frontier)
        centroids_with_utilities.append((centroid, utility))

    centroids_with_utilities.sort(key=lambda x: x[1])
    next_destination = centroids_with_utilities[0][0]
    robbie.next_coord = next_destination

    grid.centroid = robbie.next_coord
    return robbie, robbie.next_coord


def exploration_state_machine(robbie, grid):
    """
    Use frontier planning, or another exploration algorithm, to explore the grid.

    Arguments:
        robbie: instance of the robot
        grid: instance of the grid

    Returns:
        robbie: 'updated' instance of the robot

    Notes:
        Robot is considered as Point object located at the center of the traingle.
        Robot explores the map in the discretized space
        You may use the 'rrt' function (see grid.py) to find a new path whenever the robot encounters an obstacle.
        Please note that the use of rrt slows down your code, so it should be used sparingly.
        The 'get_wheel_velocities' functions is useful in setting the robot's velocities.
        You will also find many of the functions declared in 'grid.py' and 'utils.py' useful.
        Feel free to create other helper functions (in this file) as necessary.

    Alert:
        In this part, the task is to let the robot find all markers by exploring the map,
        which means using 'grid.markers' will lead  cause zero point on GraderScope.

    """

    if not hasattr(robbie, "pid_controller"):
        linear_kp = 0.002
        linear_ki = 0
        linear_kd = 0
        angular_kp = 0.02
        angular_ki = 0
        angular_kd = 0
        robbie.pid_controller = PidController(
            linear_kp, linear_ki, linear_kd, angular_kp, angular_ki, angular_kd
        )

    if (
        robbie.next_coord is None
        or grid_distance(robbie.x, robbie.y, robbie.next_coord[0], robbie.next_coord[1])
        < 0.5
    ):
        robbie, robbie.next_coord = frontier_planning(robbie, grid)

        if robbie.next_coord is None:
            if robbie.explored_cells:
                robbie.next_coord = random.choice(list(robbie.explored_cells))
            else:
                robbie.vl, robbie.vr = 0.0, 0.0
                return robbie

    path = []

    if robbie.next_coord and grid.is_collision_with_obstacles(
        (robbie.x, robbie.y), robbie.next_coord
    ):
        print((robbie.x, robbie.y))
        print("we are calling rrt")
        path = grid.rrt((robbie.x, robbie.y), robbie.next_coord)
        print(path)
        robbie.next_coord = path[1]

    robbie.vr, robbie.vl = get_wheel_velocities(robbie, robbie.next_coord)

    return robbie
