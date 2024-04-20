# Environment Setup

1. Refer to the EdStem post about [VS Code and Anaconda Setup Guide] (https://edstem.org/us/courses/50208/discussion/4105026) if you do not already have Anaconda or Miniconda installed.
2. Create a conda environment from the .yml files provided in `/environment` folder:
    - If you are running windows, use the Conda Prompt, on Mac or Linux you can just use the Terminal.
    - Navigate to the  `environment` folder (run `cd src/environment` from the root of the project)
    - Use the command: `conda env create -f ml_hw4_env.yml`
    - This should create an environment named `ml_hw4`. 
3. Activate the conda environment:
    - Windows command: `activate ml_hw4` 
    - MacOS / Linux command: `conda activate ml_hw4`

For more references on conda environments, refer to [Conda Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)