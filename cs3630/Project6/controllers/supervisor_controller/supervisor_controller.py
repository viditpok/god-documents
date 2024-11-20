from controller import Robot, Camera, Display, Supervisor
import numpy as np
import json
import math

def get_floor_dim(supervisor):

    root_node = supervisor.getRoot()

    # Get the children field of the root node
    children_field = root_node.getField('children')

    # Iterate over all children nodes to find the RectangleArena node
    for idx in range(children_field.getCount()):
        node = children_field.getMFNode(idx)
        if node.getTypeName() == 'RectangleArena':
            floor_size_field = node.getField('floorSize')
            if floor_size_field:
                floor_size = floor_size_field.getSFVec2f()
                return floor_size
    return None

def get_robot_position(supervisor):
    robot_node = supervisor.getFromDef('e-puck')  # Replace with your robot's DEF name
    if robot_node:
        translation_field = robot_node.getField('translation').getSFVec3f()
        return translation_field
    return None
    
def get_goal_position(supervisor):
    # Get the root node of the scene tree
    root_node = supervisor.getRoot()
    root_children_field = root_node.getField('children')
    
    goal_positions = []

    # Iterate through all children nodes
    for idx in range(root_children_field.getCount()):
        child_node = root_children_field.getMFNode(idx)
        
        # Check if the child node has 'wall' in its name
        if 'divergentindicator' in child_node.getTypeName().lower():  
            translation_field = child_node.getField('translation').getSFVec3f()
            print("Goal transition:", translation_field)
            goal_positions.append(translation_field)
            
    return goal_positions
    
def calculate_wall_coordinates(center, rotation, size, padding = 0.05):
    # Extract values
    cx, cy, _ = center  # Center position on the x-y plane
    width, length, _ = size  # Width and length of the wall are used in the x-y plane
    angle = rotation[3]
    # Calculate half dimensions
    half_width = width / 2.0
    half_length = length / 2.0
    
    longer_side = 'x' if width > length else 'y'
    # Define the unrotated corners relative to the center
    padding = padding

    # Define corners without padding
    corners = [
        (cx - half_width, cy - half_length),
        (cx + half_width, cy - half_length),
        (cx + half_width, cy + half_length),
        (cx - half_width, cy + half_length)
    ]

    corners_with_padding = [
        (corners[0][0] - padding, corners[0][1] - padding),
        (corners[1][0] + padding, corners[1][1] - padding),
        (corners[2][0] + padding, corners[2][1] + padding),
        (corners[3][0] - padding, corners[3][1] + padding)
    ]
    
    # Rotate the corners around the center
    rotated_corners = []
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    for x, y in corners_with_padding:
        # Translate corner to origin for rotation
        temp_x = x - cx
        temp_y = y - cy
        
        # Apply rotation
        rotated_x = temp_x * cos_angle - temp_y * sin_angle
        rotated_y = temp_x * sin_angle + temp_y * cos_angle
        
        # Translate back to original position
        rotated_x += cx
        rotated_y += cy 
        
        # Append the rotated corner to the list
        rotated_corners.append([round(rotated_x,2), round(rotated_y,2)])
    
    return rotated_corners


def get_wall_positions(supervisor, padding=0.05):
    wall_corners = []

    # Get the root node of the scene tree
    root_node = supervisor.getRoot()
    root_children_field = root_node.getField('children')
    print("Padding: ", padding)
    # Iterate through all children nodes
    for idx in range(root_children_field.getCount()):
        child_node = root_children_field.getMFNode(idx)

        # Check if the child node has 'wall' in its name
        if 'wall' in child_node.getTypeName().lower():  # Adjust the condition based on your naming convention
            # If it's a wall, get its 'translation' field
            translation_field = child_node.getField('translation')
            rotation_field = child_node.getField('rotation')
            # Additionally, get its size field assuming it exists and is named 'size'
            size_field = child_node.getField('size')
            if translation_field and size_field:
                # Retrieve the wall's position and size
                position = translation_field.getSFVec3f()
                rotation = rotation_field.getSFVec3f()
                print("angle:", rotation)
                size = size_field.getSFVec3f()  # Assuming size is [width, length, not needed]
                corners = calculate_wall_coordinates(position, rotation, size, padding)
                wall_corners.append(corners)

    return wall_corners


def point_in_rotated_rectangle(point, rectangle_vertices):
    # Calculate vectors AB, BC, CD, and DA
    AB = (rectangle_vertices[1][0] - rectangle_vertices[0][0], rectangle_vertices[1][1] - rectangle_vertices[0][1])
    BC = (rectangle_vertices[2][0] - rectangle_vertices[1][0], rectangle_vertices[2][1] - rectangle_vertices[1][1])
    CD = (rectangle_vertices[3][0] - rectangle_vertices[2][0], rectangle_vertices[3][1] - rectangle_vertices[2][1])
    DA = (rectangle_vertices[0][0] - rectangle_vertices[3][0], rectangle_vertices[0][1] - rectangle_vertices[3][1])

    # Calculate vectors AP, BP, CP, and DP
    AP = (point[0] - rectangle_vertices[0][0], point[1] - rectangle_vertices[0][1])
    BP = (point[0] - rectangle_vertices[1][0], point[1] - rectangle_vertices[1][1])
    CP = (point[0] - rectangle_vertices[2][0], point[1] - rectangle_vertices[2][1])
    DP = (point[0] - rectangle_vertices[3][0], point[1] - rectangle_vertices[3][1])

    # Calculate dot products
    dot_AB_AP = AB[0] * AP[0] + AB[1] * AP[1]
    dot_BC_BP = BC[0] * BP[0] + BC[1] * BP[1]
    dot_CD_CP = CD[0] * CP[0] + CD[1] * CP[1]
    dot_DA_DP = DA[0] * DP[0] + DA[1] * DP[1]

    # Check if point is inside rectangle
    return 0 <= dot_AB_AP <= AB[0] * AB[0] + AB[1] * AB[1] and \
           0 <= dot_BC_BP <= BC[0] * BC[0] + BC[1] * BC[1] and \
           0 <= dot_CD_CP <= CD[0] * CD[0] + CD[1] * CD[1] and \
           0 <= dot_DA_DP <= DA[0] * DA[0] + DA[1] * DA[1]

def check_if_in_wall(curr_x, curr_y, walls):
    for wall in walls:
        if point_in_rotated_rectangle([curr_x, curr_y], wall):
            return True
    return False
            
def check_if_goal(curr_x, curr_y, goal_position, threshold = 0.01):
    for goal_x, goal_y, _ in goal_position:
        if goal_x - threshold <= curr_x <= goal_x + threshold and \
            goal_y - threshold <= curr_y <= goal_y + threshold:
            return True
    return False    

def check_if_in_circle(curr_x, curr_y, goal_position, radius = 0.05):
    for goal_x, goal_y, _ in goal_position:
        distance = math.sqrt((curr_x - goal_x)**2 + (curr_y - goal_y)**2)
        if distance <= radius:
            return True
    return False

def generate_map_of_grid_size(walls, width, height, goal_position, grid_size = 0.2):
    
    layout = []
    
    curr_x, curr_y = -1*width/2, 1*height/2
    
    while curr_y >= -1 * height/2:
        row = []
        while curr_x <= width/2:
            if check_if_in_wall(curr_x, curr_y, walls) or \
                    check_if_in_wall(curr_x + grid_size, curr_y, walls) or \
                    check_if_in_wall(curr_x, curr_y - grid_size, walls) or \
                    check_if_in_wall(curr_x + grid_size, curr_y - grid_size, walls):
                row.append("O")
            # elif check_if_goal(curr_x, curr_y, goal_position):
            #     row.append("U")
            elif check_if_in_circle(curr_x, curr_y, goal_position):
                row.append("U")
            else:
                row.append(".")
            curr_x += grid_size
        layout.append(row)
        curr_x = -1*width/2
        curr_y -= grid_size
    
    return layout
    
def generate_map_json(robot_position, walls, width, height, goal_position, layout, grid_size, world_title):
    
    print(robot_position)
    
    map_data = {
        "width": int(width/grid_size),
        "height": int(height/grid_size),
        "cont_width": width,
        "cont_height": height,
        "start": [(round(robot_position[0],2)+width/2)//grid_size, (-round(robot_position[1],2)+height/2)//grid_size,0],  # Assuming X and Z are the 2D floor coordinates
        "goals": goal_position, #[[round(goal_position[0],2), round(goal_position[1],2)]],
        "obstacles": walls, # Convert to 2D coordinates,
        "grid_size":grid_size,
        "scale": 5,    #Scaling for Tk inter GUI
        "layout": layout
    }
    
    with open("../exploration_controller/maps/{}.json".format(world_title), "w") as json_file:
        json.dump(map_data, json_file, indent=4)


supervisor = Supervisor()
root = supervisor.getRoot()
print(root)
world_info = root.getField("children").getMFNode(0)  # Assuming WorldInfo is the first child node, adjust index if necessary
world_title = world_info.getField("title").getSFString()
timestep = int(supervisor.getBasicTimeStep())

walls_RRT = get_wall_positions(supervisor, padding=0.04)
walls_grid = get_wall_positions(supervisor, padding=0.04)

robot_position = get_robot_position(supervisor)
goal_position = get_goal_position(supervisor)
floor_dim = get_floor_dim(supervisor)
height, width = floor_dim[0], floor_dim[1]

grid_size = 0.04
layout = generate_map_of_grid_size(walls_grid, width, height, goal_position, grid_size=grid_size)
generate_map_json(robot_position, walls_RRT, width, height, goal_position, layout, grid_size, world_title)

#print(layout)
print(world_title)
print("Map Generated!")
    
    
    
  