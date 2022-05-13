# Machine learning workshop series 2022

The directory contains code for each of the workshop sessions in machine learning for geotechnical engineering presented the spring 2022. We upload code to this repo after each session.

Before you start to code, make sure you have installed:

- An IDE: VSCode, Spyder, Atom, Pycharm etc.
- An package handling and coding environment system. In further coding sessions we show instructions using conda, but feel free to use other systems such as pipenv etc. Conda can be downladed either in a GUI version, using Anaconda (https://www.anaconda.com/products/distribution), or the version called miniconda (https://docs.conda.io/en/latest/miniconda.html) without GUI and tons of other stuff in Anaconda you probably don't need :-)

# Good practise for scientific development

First we would like to mention three excellent papers that describe good practise in scientific computing.

- https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510&ref=https://githubhelp.com
- https://doi.org/10.1016/j.patter.2021.100206.
- https://journals.plos.org/plosbiology/article/info:doi/10.1371/journal.pbio.1001745

## Version control system

Git is a version control system. To get the code locally on your computer. You do this only once.

1. Install git
2. On a linux or windows terminal maneuver to your project directory where you want to store different coding projects.
3. Clone repo with:

    ```bash
    git clone <url copied from repo>
    ```

Save datasets in a directory called datasets in your local version of the directory
`Machine learning workshop series for geotechnical engineering 2022`

## Environment

Use one environment for each coding project. For these 5 sessions we will use the same 
datasets and all sessions are basically a part of the same project. It is then ok with the 
the same environment for all sessions.

1. Create an environment called `ml_sessions_2022` using `environment.yaml` with the help of `conda`. If you get pip errors, install pip libraries manually, e.g. `pip install pandas`

   ```bash
   conda env create --file environment.yaml
   ```

2. Activate the new environment with:

   ```bash
   conda activate ml_sessions_2022
   ```

## Version control

We recommend to register for your own github account and make one repo for your code in 
the workshop sessions. After every session you push the code to your personal github repo.
That is a good learning task for taking care of version control!

## Before each coding session

We recommend some steps before each workshop session

1. Get the latest code in repo. You will probably be asked for a git-token for authentication:

    ```bash
    git pull
    ```

2. Update the necessary libraries and activate the environment by calling:

    ```bash
    conda env update --file environment.yaml
    conda activate ml_sessions_2022
    ```
