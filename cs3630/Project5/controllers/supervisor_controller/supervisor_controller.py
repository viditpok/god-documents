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
    goal_node = supervisor.getFromDef('goal')  # Replace with your robot's DEF name
    if goal_node:
        translation_field = goal_node.getField('translation').getSFVec3f()
        return translation_field
    return None
    
def calculate_wall_coordinates(center, rotation, size):
    # Extract values
    cx, cy, _ = center  # Center position on the x-y plane
    width, length, _ = size  # Width and length of the wall are used in the x-y plane
    angle = rotation[3]
    # Calculate half dimensions
    half_width = width / 2.0
    half_length = length / 2.0
    
    longer_side = 'x' if width > length else 'y'
    # Define the unrotated corners relative to the center
    padding = 0.05

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



def get_wall_positions(supervisor):
    wall_corners = []

    # Get the root node of the scene tree
    root_node = supervisor.getRoot()
    root_children_field = root_node.getField('children')

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
                corners = calculate_wall_coordinates(position, rotation, size)
                wall_corners.append(corners)

    return wall_corners

def generate_map_json(robot_position, walls, width, height, goal_position, world_title):
    map_data = {
        "width": width,
        "height": height,
        "start": [round(robot_position[0],2), round(robot_position[1],2)],  # Assuming X and Z are the 2D floor coordinates
        "goals": [[round(goal_position[0],2), round(goal_position[1],2)]],
        "obstacles": walls # Convert to 2D coordinates
    }
    # TO DO: generate this name dynamically based on the loaded world
    
    with open("../rrt_controller/maps/{}.json".format(world_title), "w") as json_file:
        json.dump(map_data, json_file, indent=4)


supervisor = Supervisor()
root = supervisor.getRoot()
world_info = root.getField("children").getMFNode(0)  # Assuming WorldInfo is the first child node, adjust index if necessary
world_title = world_info.getField("title").getSFString()
timestep = int(supervisor.getBasicTimeStep())

walls = get_wall_positions(supervisor)
robot_position = get_robot_position(supervisor)
goal_position = get_goal_position(supervisor)
floor_dim = get_floor_dim(supervisor)
height, width = floor_dim[0], floor_dim[1]

print("Wall positions:", walls)
print("Robot position:", robot_position)
print("Floor Dim:", height, width)
print("Goal", goal_position)

generate_map_json(robot_position, walls, width, height, goal_position, world_title)


    
    
    
  