# CS 4476 Assignment 2: SIFT Local Feature Matching

## Getting started

- See [Assignment 0](https://github.gatech.edu/vision/assignment-0) for detailed environment setup.
- Ensure that you are using the environment `cv_proj2`, which you can install using the install script `conda/install.sh`.

## Logistics

- Submit via [Gradescope](https://gradescope.com).
- Part 5 of this project is **optional**.
- Additional information can be found in `docs/proj2.pdf`.

## Rubric (100 points in total + 10 points for Extra Credit)

#### 80 pts: Coding

- +5 pts: `compute_harris_response_map()` in `part1_harris_corner.py`
- +5 pts: `compute_image_gradients()` in `part1_harris_corner.py`
- +5 pts: `nms_maxpool_pytorch()` in `part1_harris_corner.py`
- +5 pts: `get_harris_interest_points()` in `part1_harris_corner.py`
- +10 pts: `compute_normalized_patch_descriptors()` in `part2_patch_descriptor.py`
- +3 pts: `compute_feature_distances_2d()` in `part3_feature_matching.py`
- +2 pts: `compute_feature_distances_10d()` in `part3_feature_matching.py`
- +5 pts: `match_features_ratio_test()` in `part3_feature_matching.py`
- +4 pts: `get_magnitudes_and_orientations()` in `part4_sift_descriptor.py`
- +4 pts: `get_gradient_histogram_vec_from_patch()` in `part4_sift_descriptor.py`
- +4 pts: `get_feat_vec()` in `part4_sift_descriptor.py`
- +4 pts: `get_SIFT_descriptors()` in `part4_sift_descriptor.py`
- +4 pts for Feature_matching_speed in `part4_sift_descriptor.py`
- +10 pts for Feature_matching_accuracy in `part4_sift_descriptor.py`
- +4 pts: `test_rotate_image()` in `part4_sift_descriptor.py`
- +1 pt: `get_correlation_coeff()` in `part4_sift_descriptor.py`
- +5 pts: `get_intensity_based_matches()` in `part4_sift_descriptor.py`

#### 20 pts: Writeup Report

- +1 pts: page 2 (Part 1: Harris corner detector)
- +1 pts: page 3 (Part 1: Harris corner detector)
- +1 pts: page 4 (Part 1: Harris corner detector)
- +1 pts: page 5 (Part 1: Harris corner detector)
- +1 pts: page 6 (Part 1: Harris corner detector)
- +1 pts: page 7 (Part 1: Harris corner detector)
- +1 pts: page 8 (Part 2: Normalized patch feature descriptor)
- +2 pts: page 9 (Part 3: Feature matching)
- +1 pts: page 10 (Part 3: Feature matching)
- +1 pts: page 11 (Part 4: SIFT feature descriptor)
- +2 pts: page 12 (Part 4: SIFT feature descriptor)
- +1 pts: page 13 (Part 4: SIFT feature descriptor)
- +1 pts: page 14 (Part 4: SIFT feature descriptor)
- +1 pts: page 15 (Part 4: SIFT feature descriptor)
- +1 pts: page 16 (Part 4: SIFT feature descriptor)
- +1 pts: page 17 (Part 4: Intensity-based Matching)
- +1 pts: page 18 (Part 4: Intensity-based Matching)
- +1 pts: page 19 (Part 4: SIFT vs Intensity-based Matching)

### 10 pts: Extra Credit

- +2.5 pts: page 20 (Part 5: SIFT Descriptor Exploration)
- +2.5 pts: page 21 (Part 5: SIFT Descriptor Exploration)
- +2.5 pts: page 22 (Part 5: SIFT Descriptor Exploration)
- +2.5 pts: page 23 (Part 5: SIFT Descriptor Exploration)

- Will be evaluated based on your report and qualitative results. You do not need to upload your code separately.

##### -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format

## Submission format

This is very important as you will lose 5 points for every time you do not follow the instructions.

1. Generate the zip folder (`<your_gt_username>.zip`) for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`. It should contain:
    - `src/`: directory containing all your code for this assignment
    - `setup.cfg`: setup file for environment, do not need to change this file
    - `additional_data/`: the images you took for the extra credit section Part 5, and/or if you use any data other than the images we provide, please include them here
    - `README.txt`: (optional) if you implement any new functions other than the ones we define in the skeleton code (e.g., any extra credit implementations), please describe what you did and how we can run the code. We will not award any extra credit if we can't run your code and verify the results.
2. `<your_gt_username>_proj2.pdf` - your report


## Important Notes

- Please follow the environment setup in [Assignment 0](https://github.gatech.edu/vision/assignment-0).
- Do **not** use absolute paths in your code or your code will break.
- Use relative paths like the starter code already does.
- Failure to follow any of these instructions will lead to point deductions. Create the zip file by clicking and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).
