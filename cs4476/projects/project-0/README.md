# Welcome to CS 4476!

This is an **ungraded** assignment will help you get up and running with a working environment! For those that are familiar with this setup -- let's rehash anyways.
We'll focus on install a few crucial tools to help us maintain consistent environments between students.

- [Visual Studio Code](https://code.visualstudio.com/Download)
- [GitHub CLI](https://cli.github.com/)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Visual Studio Code

[Visual Studio Code](https://code.visualstudio.com/Download) is **recommended** for this course for a few reasons.

- Consistent development environment (so the teaching staff can more easily help!)
- Ease of integration with Jupyter Notebooks and `git`.

### macOS Users (Optional)

`brew` is a very useful package management tool for `macOS`. It can make some of the above installation even simpler.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install --cask visual-studio-code
brew install gh
```

## Visual Studio Code Extensions

Download the following extensions:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

*Note: confirm that all the extensions are installed/enabled before continuing. 

## Why `git`?

[git](https://git-scm.com/) is an open source, version-control tool. It acts as a history or "snapshot" of your work over time. We **highly** recommend using version control for one core reason:

- Prevents loss of data

Consider committing your changes after major assignment milestones so you do not lose your work. However, do **not** post your code publicly on the Internet, in a public repository or otherwise.

### Authenticating with GT Github Enterprise

Let's setup authentication to our internal GitHub instance.

```bash
gh auth login --hostname github.gatech.edu --web
```

This will prompt you with:

```bash
- Press Enter to open github.gatech.edu in your browser... 
```

Click `enter` and you should be good to go (or will be asked to log into GT Github Enterprise on your browser)!

## Template Repositories

Templates are analagous to a "private fork" of a repository. You will be **required** to keep these repositories private (it's the default option).

- Click the green "Use this Template" button and create your repository.
- Name your repository `assignment-0` and make sure it's **private**.

## Cloning your Repository

From the VSCode terminal, traverse to the directory where you store your code. Enter the following:

```bash
gh repo clone github.gatech.edu/[YOUR_USERNAME]/assignment-0
```

## Environment Setup

We will be using Miniconda and pip to create a consistent Python environment for running and testing our code. For each assignment, we will create a new conda environment with slightly different packages (specified in `requirements.txt`). You will need to install Miniconda before you can start working in these environments.

See the [Miniconda Installation Instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install the correct version of Miniconda for your machine and to troubleshoot any installation issues. See the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) for more information on system requirements and what Miniconda does.

## Setting up assignment 0

In VSCode, open your cloned template repository. In the `conda` folder, you will see an installation script, `install.sh`. Running this script will create your `conda` environment, which should then be activated. 

If you do not see `cv_proj0` at the start of each line of your VSCode command prompt, you will need to manually activate your environment with the following commands from the root of your repository.

```bash
conda init
conda activate cv_proj0
pip install -e .
```

**Important Note**: You will need to select a `Default Interpreter` usually at the bottom right of your VSCode window for linting / static analysis to work properly.

- Please click the `Select Interpreter` at the bottom right of the VSCode window and select the environment you just created.
- You may need to restart VSCode for the new environment to show up in this list.

## Unit Test Setup

We will use `pytest` in this class as a testing framework. This will run unit tests on your code to verify correctness. Some of the unit tests will given to you as a courtesy; however, others will be run at submission time.

### Run the unit tests

The following command will run all unit tests in the `tests` folder.

```bash
pytest tests
```

The unit tests will initially fail -- modify `src/vision/linalg.py` and see if you can pass the tests!

## Jupyter Notebook

Each assignment (except this one) will have a jupyter notebook which will allow you to run and test your code in an interactive Python coding environment. Jupyter notebook should already be installed, but if not you can visit [this page](https://jupyter.org/install) to install.

You can open these notebooks directly in VSCode or use the following command to start a jupter notebook session:
```bash
jupyter notebook ./<file_name>.ipynb
```

Once in the notebook, you can edit text cells by double-clicking on them and edit/run code cells (use the shortcut cmd+enter or shift+enter to run code cells).

## Submission
To create the zip file to upload on gradescope, run the following from within your conda environment.

```
python submission.py --gt_username <your_username>
```

# Conclusion

We hope this assignment helped you get your environment setup for this course. If you have any questions / concerns, please post on [Ed](https://edstem.org/us/courses/52650/).
