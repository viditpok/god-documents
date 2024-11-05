from rrt import *
import json
import os
from robot_sim import DDRobot
import random
import pickle

def is_path_collision_free(map, smooth_path):
    if len(smooth_path) > 0:
        for i in range(1, len(smooth_path)):
            if map.is_collision_with_obstacles(((smooth_path[i-1], smooth_path[i]))):
                print("Collided")
                return False
    
    if (smooth_path[0].x, smooth_path[0].y) != (map.get_start().x, map.get_start().y):
        print("Start point not matching")
        return False
    
    if get_dist(smooth_path[-1], map.get_goals()[0]) > 1:
        print("Goal point not matching")
        return False
    
    return True
    
def test_path_smoothing(map, path):
    print(len(path))
    smooth_path = map.compute_smooth_path(path,10)
    print(len(smooth_path))
    if is_path_collision_free(map, smooth_path):
        return True, len(smooth_path)
    return False, None


def is_solution_valid_from_path(map, path):

    if len(path) > 0:
        for i in range(1, len(path)):
            node0 = Node(tuple(path[i-1]))
            node1 = Node(tuple(path[i]))
            if map.is_collision_with_obstacles(((node0, node1))):
                print("Collided here")
                return False
            
    with open(map.fname) as configfile:
        # Load dimensions from json file
        config = json.loads(configfile.read())
        start_point = Node(tuple(config['start']))

    if (path[0][0], path[0][1]) != (start_point.x, start_point.y):
        print("Start point not matching")
        return False
    
    if get_dist(Node(tuple(path[-1])), map.get_goals()[0]) > 1:
        print("Goal point not matching")
        return False

    return True

class GradingThread():
    """RRT grading thread
    """
    def __init__(self, maps, path_files=None):
        self.maps = maps
        self.path_files = path_files

    def run(self):
        print("Grader running...\n")
        points = 0
        total = len(self.maps) * 20

        for map in self.maps:
            cmap = Map(map)
            RRT(cmap)
            if (cmap.is_solution_valid()):
                points += 20
            print(map + ": " + str(points) + "/" + str(total) + " points")

        print("\nScore = " + str(points) + "/" + str(total) + "\n")
    
    def check_path_smoothing(self):
        print("Grader running...\n")
        points = 0
        total = len(self.maps) * 20
        node_count_dict = {}
        node_count_dict["maps/maze3.json"] = 6 # max path length after path smoothing - set as x1.5 of TAs' implementation
        


        for map, path_file in zip(self.maps, self.path_files):
            path_file = path_file
            map_file = map
            with open(path_file, "rb") as file:
                loaded_path = pickle.load(file)
            cmap = Map(map_file)
            min_path_length = sys.maxsize

            for i in range(5):
                isValid, count = test_path_smoothing(cmap, loaded_path)
                if isValid:
                    min_path_length = min(min_path_length, count)
            
            if min_path_length <= node_count_dict[map]:
                points += 20
            
            map_name = os.path.basename(map_file)
            print(map_name + ": " + str(points) + "/" + str(total) + " points")
        
        print("\nScore = " + str(points) + "/" + str(total) + "\n")
        return points

    def check_inbound(self):
        print("Testing check_inbound...\n")
        nodes = [Node((random.uniform(2.5, 3.0),random.uniform(2.0, 3.0))),
                 Node((random.uniform(1.8, 2.7),-random.uniform(0.5, 1.8))),
                 Node((random.uniform(0.8, 1.3),-random.uniform(0.0, 0.9)))]
        checks = []
        points = 0
        cmap = Map(self.maps[0])
        total = len(self.maps) * 5
        for node in nodes:
            checks.append(cmap.is_inbound(node))
        if checks == [False,False,True]:
            points += 5
        else:
            print("is_inbound function not correctly implemented")
        print("\nScore = " + str(points) + "/" + str(total) + "\n")

    def check_insideobstacle(self):
        print("Testing check_inside_obstacle...\n")
        node = Node((-0.69,-0.5))
        cmap = Map(self.maps[0])
        points = 0
        total = len(self.maps) * 5
        if cmap.is_inside_obstacles(node):
            points += 5
        else:
            print("is_inside_obstacles function not correctly implemented")
        print("\nScore = " + str(points) + "/" + str(total) + "\n")

    def check_stepfromto(self):
        print("Testing step_from_to()...\n")
        node1 = Node((50,50))
        node2 = Node((275,500))
        cmap = Map(self.maps[0])
        points = 0
        total = len(self.maps) * 10
        ans = cmap.step_from_to(node1,node2)
        if math.isclose(ans.x,83.54101966249685, abs_tol =0.01) and math.isclose(ans.y,117.08203932499369, abs_tol =0.01):
            points += 10
        else:
            print("step from to function not correctly implemented")
        print("\nScore = " + str(points) + "/" + str(total) + "\n")


if __name__ == "__main__":
    tests = {}
    if len(sys.argv) > 1:
        try:
            # Test to run
            test = str(sys.argv[1]) # Give {test1, test2, test3}
        except:
            print("Error: please give the test option <helpers/rrt/smoothing>")
            raise
    else:
        print("correct usage: python3 autograder.py <helpers/rrt/smoothing>")
        exit()

    if test=="helpers":
        maps = ["maps/maze1.json"]
        grader = GradingThread(maps)
        grader.check_inbound()
        grader.check_insideobstacle()
        grader.check_stepfromto()

    if test == "rrt":
        maps = ["maps/maze1.json", "maps/maze2.json", "maps/maze3.json"]
        grader = GradingThread(maps)
        grader.run()
    
    if test == "smoothing":
        maps = ["maps/maze3.json"]
        path_files = ["paths/maze3_path.pkl"]
        grader = GradingThread(maps, path_files=path_files)
        grader.check_path_smoothing()


