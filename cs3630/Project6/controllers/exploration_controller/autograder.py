import sys
from grid import Grid
from robot import Robot_Sim
from robot_gui import RobotEnv, RobotEnvThread
from concurrent.futures import ThreadPoolExecutor
import re

def grade(maps, program_state, time_limit, point_per_map):
    print("Grader running...\n")
    points = 0

    threads = []

    for map in maps:
        grid = Grid(map)
        robot_init_pose = grid.start
        robbie = Robot_Sim(*robot_init_pose)
        robot_env = RobotEnv(robbie, grid, testing=True)
        robot_env.program_state = program_state
        robot_thread = RobotEnvThread(robot_env, testing=True, time_limit=time_limit)
        threads.append(robot_thread)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(thread.run) for thread in threads]
        results = [f.result() for f in futures]
        for i, result in enumerate(results):
            marker_list, num_total_markers = result
            point = point_per_map * (len(marker_list) / num_total_markers)
            print(maps[i] + ": " + str(round(point,2)) + "/" + str(point_per_map) + " points")
            points += point
        print("\nScore = " + str(points) + "/" + str(float(point_per_map * len(results))) + "\n")
            
if __name__ == "__main__":
    
    maps = ["maps/maze1.json", "maps/maze2.json", "maps/maze3.json"]
    
    time_limit = 120

    if len(sys.argv) > 1:
        test = "exploration"
        if len(sys.argv) >= 2:
            maps = [sys.argv[1]]
    else:
        print("Correct usage: python3 autograder.py [map_filepath]")
        exit(1)

    ## Check for use of grid.markers() ##
    string_to_check = 'grid.markers'

    with open('exploration.py', 'r') as file:
        content = file.read()

    # Use regular expression to find the usage of the member variable #
    pattern = re.compile(re.escape(string_to_check))
    matches = pattern.findall(content)

    pattern = re.compile(re.escape(f"'{string_to_check}'"))
    doc_matches = pattern.findall(content)

    if (len(doc_matches) == 1 and len(matches) > 1) or (len(doc_matches) == 0 and len(matches) > 0):
        print(f"'{string_to_check}' present in the file. Invalid submission for exploration.py")
    else:
        if test == "exploration":
            grade(maps, test, time_limit, 12.5)

