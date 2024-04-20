# CS-4476 Assignment 1: Convolution and Hybrid Images

## Getting Started

- Check Assignment 0 for environment installation.
- Ensure that you are using the environment `cv_proj1`, which you can install using the install script `conda/install.sh`.

## Logistics

- Submit via [Gradescope](https://gradescope.com).
- Detailed instructions can be found in `docs/project-1.pdf`.

## Rubric (100 pts in total)

#### 80 pts: Coding

- +5 pts: `create_Gaussian_kernel_1D()` in `part1.py`
- +5 pts: `create_Gaussian_kernel_2D()` in `part1.py`
- +5 pts: `separate_Gaussian_kernel_2D()` in `part1.py`
- +12.5 pts: `my_conv2d_numpy()` in `part1.py`
- +7.5 pts: `create_hybrid_image()` in `part1.py`
- +5 pts: `make_dataset()` in `part2_datasets.py`
- +5 pts: `get_cutoff_frequencies()` in `part2_datasets.py`
- +5 pts: `__len__()` in `part2_datasets.py`
- +5 pts: `__getitem__()` in `part2_datasets.py`
- +5 pts: `get_kernel()` in `part2_models.py`
- +5 pts: `low_pass()` in `part2_models.py`
- +10 pts: `forward()` in `part2_models.py`
- +5 pts: `my_conv2d_pytorch()` in `part3.py`

#### 20 pts: Writeup Report

- +1 pts: page 2 (Part 1: Gaussian Kernels)
- +1 pts: page 3 (Part 1: Image filtering)
- +1 pts: page 4 (Part 1: Image filtering)
- +1 pts: page 5 (Part 1: Image filtering)
- +1.5 pts: page 6 (Part 1: Hybrid images)
- +1.5 pts: page 7 (Part 1: Hybrid images)
- +1.5 pts: page 8 (Part 1: Hybrid images)
- +1.5 pts: page 9 (Part 2: Hybrid images with PyTorch)
- +1.5 pts: page 10 (Part 2: Hybrid images with PyTorch)
- +1.5 pts: page 11 (Part 2: Hybrid images with PyTorch)
- +2 pts: page 12 (Part 3: Understanding input/output shapes in PyTorch)
- +1 pts: page 13 (Part 3: Understanding input/output shapes in PyTorch)
- +1 pts: page 14 (Part 3: Understanding input/output shapes in PyTorch)
- +1 pts: page 15 (Part 3: Understanding input/output shapes in PyTorch)
- +1 pts: page 16 (Part 3: Understanding input/output shapes in PyTorch)
- +1 pts: page 17 (Conclusion)

##### -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format

## Submission format

This is very important as you will lose 5 points for every time you do not follow the instructions.

1. Generate the zip folder (`<your_gt_username>.zip`) for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`. It should contain:
    - `src/`: directory containing all your code for this assignment
    - `cutoff_frequency.txt`: .txt file containing the best cutoff frequency values you found for each pair of images in `data/`
    - `setup.cfg`: setup file for environment, do not need to change this file
    - `additional_data/`: (optional) if you use any data other than the images we provide, please include them here
    - `README.txt`: (optional) if you implement any new functions other than the ones we define in the skeleton code (e.g., any helper function implementations), please describe what you did and how we can run the code
2. `<your_gt_username>_proj1.pdf` - your report


## Important Notes

- Please follow the environment setup in Assignment 0.
- Do **not** use absolute paths in your code or your code will break.
- Use relative paths like the starter code already does.
- Failure to follow any of these instructions will lead to point deductions. Create the zip file by clicking and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).

## Project Structure

```console
.
├── README.md
├── cutoff_frequencies.txt
├── data
│   ├── 1a_dog.bmp
│   ├── 1b_cat.bmp
│   ├── 2a_motorcycle.bmp
│   ├── 2b_bicycle.bmp
│   ├── 3a_plane.bmp
│   ├── 3b_bird.bmp
│   ├── 4a_einstein.bmp
│   ├── 4b_marilyn.bmp
│   ├── 5a_submarine.bmp
│   └── 5b_fish.bmp
├── docs
│   └── report.pptx
│   └── project_1.pdf
├── project-1.ipynb
├── pyproject.toml
├── zip_submission.py
├── README.md
├── setup.cfg
├── src
│   └── vision
│       ├── __init__.py
│       ├── part1.py
│       ├── part2_datasets.py
│       ├── part2_models.py
│       ├── part3.py
│       └── utils.py
└── tests
    ├── __init__.py
    ├── __pycache__
    ├── test_part1.py
    ├── test_part2.py
    ├── test_part3.py
    └── test_utils.py
```
