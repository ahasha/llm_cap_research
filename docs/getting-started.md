# Getting started

:::{tip}
Customize this documentation page to describe how to get the project set up
from a clean install, including the commands necessary to get the raw data and
then how to make the cleaned, final data sets, models, and analysis.

Generic instructions for installing and utilizing the features of the
cookiecutter template are already included below.
:::

## Software Prerequisites

This project uses [poetry](https://python-poetry.org/docs/) to recreate an identical analytical software environment on each developer's machine.  To bootstrap this environment, you will need an existing installation of

* Python version 3.11
* Poetry v1.40+

These can be installed using whatever tool you generally use for managing your Python environment, such as `conda` or `vitualenv`. Poetry will run from this parent environment and create an independent, reproducible python environment for running the model development code.

## Environment set up

Begin by cloning the model development repository

```bash
$ git clone git@github.com:ahasha/llm_cap_research.git
```

and change to the root directory of the project.  The command

```bash
$ make initialize
```

will install the project's software envionrment using poetry and install the project's git hooks, which enforce consistent code style, linting, and [dvc actions](https://dvc.org/doc/command-reference/install#description) to ensure data and code versions are synchronized.

You then run

```bash
$ poetry shell
```

to enter the virtual environment associated with the project.  Typing `exit` will exit the poetry shell, analogous to `deactivate` for a virtual environment.

## Get project data

This project uses [DVC](https://dvc.org) to store and track versioned data.  Similar to how git works, the DVC cache is a hidden storage folder (by default in
`.dvc/cache`) containing all versions of all files and directories tracked by
DVC.  It uses a [content-addressable structure](https://dvc.org/doc/user-guide/project-structure/internal-files#structure-of-the-cache-directory) that allows only the current version of tracked data corresponding to the current state of the code in the git repository to be automatically loaded into the workspace.

A shared dvc cache (analogous to a shared git remote on github.com) is located
in Google Cloud storage at `gs://hasha-ds-portfolio-projects/llm_cap_research/`.

As long as you can access this cloud bucket from your current working environment,
you can populate your local cache (analogous to running `git clone` to get the latest
version of a codebase) by running

```bash
$ dvc pull
```

from any directory inside the project.

## Troubleshooting

Poetry can be a bit finicky to install and configure if you haven't used it before.

The [official installation instructions](https://python-poetry.org/docs/#installing-with-the-official-installer) call for installing in a new virtual environment to isolate it from the rest of your system.

> This ensures that dependencies will not be accidentally upgraded or uninstalled, and allows Poetry to manage its own environment.

It's definitely my experience that just running `pip install poetry` in an existing
environment leads to weird cross-contamination problems when trying to build new
environments with poetry.  You need to create a clean "boostrap" environment with
Python 3.11 and poetry installed, and avoid using this
environment for other development tasks.

You can follow the official poetry install instructions to achieve this.  Another approach
that worked for me was to set up the bootstrap environment using [`mamba`](https://mamba.readthedocs.io/en/latest/installation.html).