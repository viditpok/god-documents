import os, sys
from vision_only_calculation import vision_only_depth_calculation
import re
import numpy as np
import math
import cv2


class Tests:

    def __init__(self):
        path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.vision_data_path = os.path.join(path, "Vision_only")
        self.vision_lidar_data_path = os.path.join(path, "Vision_Lidar")
        vision_solution = os.path.join(path, "Vision_only/outputs.txt")
        lidar_solution = os.path.join(path, "Vision_Lidar/solution.txt")
        with open(vision_solution, "r") as file:
            file_content = file.read()
        results = self.extract_distance_and_angle_from_text(file_content)
        self.vision_solution = {}
        for i, (distance, angle) in enumerate(results, start=0):
            self.vision_solution[i] = {}
            self.vision_solution[i]["depth"] = distance
            self.vision_solution[i]["angle"] = angle

        with open(lidar_solution, "r") as file:
            file_content = file.read()
        lidar_data, results = self.extract_lidar_sol(file_content)
        self.lidar_sol = {}
        for i, (lidar, (dist, angle)) in enumerate(zip(lidar_data, results), start=0):
            self.lidar_sol[i] = {}
            self.lidar_sol[i]["lidar_data"] = lidar
            self.lidar_sol[i]["dist"] = dist
            self.lidar_sol[i]["angle"] = angle

    def extract_distance_and_angle_from_text(self, text):
        pattern = r"Depth, Angle calculation using vision only : \((\d+(\.\d+)?), (-?\d+(\.\d+)?)\)"
        matches = re.findall(pattern, text)
        results = [(float(match[0]), float(match[2])) for match in matches]
        return results

    def extract_lidar_sol(self, file_content):
        distance_pattern = re.compile(
            r"Distance, Angle calculation using vision \+ lidar : \((\d+\.\d+), (-?\d+\.\d+)\)"
        )
        matches = re.findall(distance_pattern, file_content)
        results = [(float(match[0]), float(match[1])) for match in matches]
        lidar_arrays = []
        lidar_matches = re.finditer(r"lidar_array:\n\[(.*?)\]", file_content, re.DOTALL)

        for match in lidar_matches:
            lidar_content = match.group(1)
            lidar_array = [
                float(value) if value != "\tmath.inf" else math.inf
                for value in lidar_content.split(",")
            ]
            lidar_arrays.append(lidar_array)

        return lidar_arrays, results

    def check_dist_accuracy(self, s_depth, sol, islidar=False):

        if islidar:
            scores = np.array([2, 1.5, 1, 0.5])
            err = np.array([0.05, 0.1, 0.15, 0.20]) * sol
            len_ = len(scores)
        else:
            scores = np.array([5, 4, 3, 2, 1])
            err = np.array([0.05, 0.10, 0.15, 0.2, 0.25]) * sol
            len_ = len(scores)

        for i in range(len_):
            if s_depth >= (sol - err[i]) and s_depth <= (sol + err[i]):
                return scores[i]
        return 0

    def check_angle_accuracy(self, s_angle, sol, islidar=False):
        if islidar:
            scores = np.array([8, 6, 4, 2])
        else:
            scores = np.array([5, 3.75, 2.5, 1.25])

        err = np.array([5, 10, 15, 20])

        for i in range(len(err)):
            if s_angle >= (sol - err[i]) and s_angle <= (sol + err[i]):
                return scores[i]
        return 0

    def test_vision_only_func(self) -> None:
        # get image data

        image1_data_path = [
            "Vision_only_world1_img_l.png",
            "Vision_only_world2_img_l.png",
        ]
        image2_data_path = [
            "Vision_only_world1_img_r.png",
            "Vision_only_world2_img_r.png",
        ]

        fov = 0.84
        translation = 0.06

        score_depth = 0
        score_angle = 0

        for i in range(2):
            print(f"Testing image pair {i+1}")

            # Load images
            image1 = cv2.imread(
                os.path.join(self.vision_data_path, image1_data_path[i])
            )
            image2 = cv2.imread(
                os.path.join(self.vision_data_path, image2_data_path[i])
            )

            # Calculate depth and angle
            student_depth, student_angle = vision_only_depth_calculation(
                image1, image2, fov, translation
            )

            # Print debugging information
            print(
                f"Calculated depth: {student_depth}, Expected depth: {self.vision_solution[i]['depth']}"
            )
            print(
                f"Calculated angle: {student_angle}, Expected angle: {self.vision_solution[i]['angle']}"
            )

            # Check accuracy
            depth_score = self.check_dist_accuracy(
                student_depth, self.vision_solution[i]["depth"]
            )
            angle_score = self.check_angle_accuracy(
                student_angle, self.vision_solution[i]["angle"]
            )

            # Print individual scores
            print(f"Depth score for image {i+1}: {depth_score}")
            print(f"Angle score for image {i+1}: {angle_score}")

            score_depth += depth_score
            score_angle += angle_score

        total_score = score_depth + score_angle
        print("Total score: " + str(total_score) + " out of 20")
        return total_score


if __name__ == "__main__":
    tests = Tests()
    print(tests.test_vision_only_func())
