import unittest
import math
import os
from geometry import SE2, Point
from environment import Environment
from setting import *


class TestSE2(unittest.TestCase):
    def test_transform_point_case1(self):
        pose = SE2(1, 0, 0)
        point = Point(1, 0)
        transformed_point = pose.transform_point(point)
        self.assertAlmostEqual(transformed_point.x, 2)
        self.assertAlmostEqual(transformed_point.y, 0)

    def test_transform_point_case2(self):
        pose = SE2(-1, -1, math.pi/2)
        point = Point(0, 1)
        transformed_point = pose.transform_point(point)
        self.assertAlmostEqual(transformed_point.x, -2)
        self.assertAlmostEqual(transformed_point.y, -1)

    def test_compose_case1(self):
        pose1 = SE2(1, 0, -math.pi/2)
        pose2 = SE2(1, 0, math.pi)
        pose_compose = pose1.compose(pose2)
        expected_compose = SE2(1, -1, math.pi/2)
        self.assertAlmostEqual(pose_compose.x, expected_compose.x)
        self.assertAlmostEqual(pose_compose.y, expected_compose.y)
        self.assertAlmostEqual(pose_compose.c, expected_compose.c)
        self.assertAlmostEqual(pose_compose.s, expected_compose.s)

    def test_inverse_case1(self):
        pose = SE2(1, 0, -math.pi/2)
        pose_inverse = pose.inverse()
        expected_inverse = SE2(0, -1, math.pi/2)
        self.assertAlmostEqual(pose_inverse.x, expected_inverse.x)
        self.assertAlmostEqual(pose_inverse.y, expected_inverse.y)
        self.assertAlmostEqual(pose_inverse.c, expected_inverse.c)
        self.assertAlmostEqual(pose_inverse.s, expected_inverse.s)


class TestEnvironment(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEnvironment, self).__init__(*args, **kwargs)
        config_file_path = os.path.join(CONFIG_PATH, "config_example.json")
        self.env = Environment(config_file_path)

    def test_read_marker_measures_case1(self):
        pose = SE2(0.1, 0.3, math.pi/2)
        marker_measures = self.env.read_marker_measures(pose)
        self.assertEqual(len(marker_measures), 1)
        marker_measure = marker_measures[0]
        self.assertAlmostEqual(marker_measure.depth, 0.64)
        self.assertAlmostEqual(marker_measure.angle, -0.30288486837497136)
        self.assertAlmostEqual(marker_measure.lidar_range, 0.7280109889280517)

    def test_read_marker_measures_case2(self):
        pose = SE2(0.1, -0.3, 1.2)
        marker_measures = self.env.read_marker_measures(pose)
        self.assertEqual(len(marker_measures), 2)
        marker_measures.sort(key = lambda m: m.angle)
        
        self.assertAlmostEqual(marker_measures[0].depth, 1.2634038010139386)
        self.assertAlmostEqual(marker_measures[0].angle, -0.3429514504068128)
        self.assertAlmostEqual(marker_measures[0].lidar_range, 1.398177385026664)

        self.assertAlmostEqual(marker_measures[1].depth, 1.224122362652729)
        self.assertAlmostEqual(marker_measures[1].angle, 0.22847931369400815)
        self.assertAlmostEqual(marker_measures[1].lidar_range, 1.3152946437965904)

    def test_diff_drive_kinematics_case1(self):
        omega_l, omega_r = 0.2, 0.5
        v_x, omega = self.env.diff_drive_kinematics(omega_l, omega_r)
        self.assertAlmostEqual(0.014, v_x)
        self.assertAlmostEqual(0.06, omega)

    def test_diff_drive_kinematics_case2(self):
        omega_l, omega_r = -0.2, 0.5
        v_x, omega = self.env.diff_drive_kinematics(omega_l, omega_r)
        self.assertAlmostEqual(0.006, v_x)
        self.assertAlmostEqual(0.14, omega)

if __name__ == '__main__':
    unittest.main()
