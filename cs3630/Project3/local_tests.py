from utils import *
from particle_filter import *
from grid import *

import setting
import unittest

class TestUtilFunctions(unittest.TestCase):

    def test_grid_distance(self):
        result = grid_distance(2, 1, 4, 5)
        self.assertEqual(result, math.sqrt(20))
        print("Good job! Sanity check for grid_distance passed.")

    def test_rotate_point(self):
        x,y = rotate_point(10,10,60)
        self.assertEqual((round(x,2),round(y,2)),(-3.66,13.66))
        print("Good job! Sanity check for rotate_point passed.")

    def test_add_gaussian_noise(self):
        iter=1000
        sum=0
        for i in range(iter):
            sum+=add_gaussian_noise(10,1)
        result=sum/iter
        self.assertAlmostEqual(result,10,delta=0.5)
        print("Good job! Sanity check for add_gaussian_noise passed.")

class TestParticleFilterFunctions(unittest.TestCase):
    def setUp(self):
        # mock grid and particles for testing
        self.grid = CozGrid(r"map_test.json") 
        self.old_particles = [Particle(1, 1, 0), Particle(2, 2, 0)]
        self.odometry_measurement = (1, 1, math.radians(90))
        self.robot_marker_list = [(1, 1, 0), (2, 2, 0)]
        self.particle_marker_list = [(1, 1, 0), (2.1, 2.1, 0)]

        # expected positions after motion update
        self.expected_positions = [
            (2.0, 2.0, 1.5707963267948966), (3.0, 3.0, 1.5707963267948966)
        ]
        
        # Expected pairs based on the robot and particle markers
        # Uses two lists to account for asymmetric pairs
        self.expected_pairs_p_r = [
            ((1, 1, 0), (1, 1, 0)),
            ((2.1, 2.1, 0), (2, 2, 0)),
        ]
        self.expected_pairs_r_p = [
            ((1, 1, 0), (1, 1, 0)),
            ((2, 2, 0), (2.1, 2.1, 0))
        ]
        
        # Expected likelihoods for specific pairs
        self.expected_likelihoods = {
            ((1, 1, 0), (1, 1, 0)): 1.0,
            ((2, 2, 0), (2.1, 2.1, 0)): 0.961,
        }

        # Expected particle likelihood
        self.expected_particle_likelihood = 0.961

    def test_create_random(self):
        count = 1000
        particles = create_random(count, self.grid)

        # check that list is of expected length
        self.assertEqual(len(particles), count)
        
        # check that particles are in free space
        for particle in particles:
            self.assertTrue(self.grid.is_free(particle.x, particle.y))

        print("Good job! Sanity check for create_random passed.")
    
    def test_motion_update(self):
        new_particles = motion_update(self.old_particles, self.odometry_measurement, self.grid)
        self.assertEqual(len(new_particles), len(self.old_particles))
        
        # check the positions of the new particles
        for new_particle, expected in zip(new_particles, self.expected_positions):
            self.assertAlmostEqual(new_particle.x, expected[0], delta=3*setting.ODOM_TRANS_SIGMA)
            self.assertAlmostEqual(new_particle.y, expected[1], delta=3*setting.ODOM_TRANS_SIGMA)
            self.assertAlmostEqual(new_particle.h, expected[2], delta=3*setting.ODOM_HEAD_SIGMA)
        
        print("Good job! Sanity check for motion_update passed.")

    def test_generate_marker_pairs(self):
        pairs = generate_marker_pairs(self.robot_marker_list, self.particle_marker_list)

        # check that each generated pair matches the expected pairs
        matches_pr = 0
        for expected in self.expected_pairs_p_r:
            matches_pr += 1 if expected in pairs else 0
        matches_rp = 0
        for expected in self.expected_pairs_r_p:
            matches_rp += 1 if expected in pairs else 0
        self.assertTrue(matches_pr == len(self.expected_pairs_p_r) or matches_rp == len(self.expected_pairs_r_p))

        # check that asymmetric list behaves correctly
        pairs = generate_marker_pairs(self.robot_marker_list, [])
        self.assertEqual(pairs, [])

        print("Good job! Sanity check for generate_marker_pairs passed.")

    def test_marker_likelihood(self):
        # check that likelihoods match expected likelihood for marker pairs
        for (robot_marker, particle_marker), expected_likelihood in self.expected_likelihoods.items():
            likelihood = marker_likelihood(robot_marker, particle_marker)
            self.assertAlmostEqual(likelihood, expected_likelihood, places=3)

        print("Good job! Sanity check for marker_likelihood passed.")

    def test_particle_likelihood(self):
        # get likelihoods
        likelihood = particle_likelihood(self.robot_marker_list, self.particle_marker_list)
        self.assertAlmostEqual(likelihood, 0.961, places=3)

        # check return value for no marker pairs
        empty_likelihood = particle_likelihood([], [])
        self.assertEqual(empty_likelihood, 0)

        print("Good job! Sanity check for particle_likelihood passed.")


if __name__ == '__main__':
    unittest.main()
