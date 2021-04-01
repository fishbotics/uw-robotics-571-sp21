# CSE 571 Robotics

# Important Note:
PLEASE DO NOT make a public fork of this codebase. We use it for homework, so if the solutions got out, it would be bad. We are very serious about this. In the real-world of software engineering, it's great to share your code with the world, but in the wacky world of academia, we need to ensure that you are really learning the material and not just copying from Github and/or classmates. It's great if you want to collaborate and learn from other students, but you must do your own work and write your own code for this class. If you absolutely _must_ use GitHub for some reason (for example, if you're using multiple computers and you don't want to just share the directory), you can set up a private repo and push your code there. We won't be able to see the repo and neither will other students.

To make the rules clear: If we find that you posted your solutions on Github, we will give you an automatic 50% for this entire assignment, including the math section. if we determine that you have copied code from another student or from a Github/Gitlab repository, you will get an automatic 0% for this entire assignment, including the math section.


## What the heck is this codebase? ##

Remember how we said it's great to share your code with the world when its not homework? This assignment is based off the first assignment from the UW deep learning class, so if you've taken that class, this may look familiar. In modern robotics, we use a ton of deep learning to solve complex tasks and it's an essential tool in any roboticist's toolkit. In this assignment, you're going to build out a basic neural network. We'll take care of some of the more boring parts like loading data, stringing things together, etc. so you can focus on the important parts, namely how a Neural network actually **works**.

We will be implementing everything using [Numpy](https://docs.scipy.org/doc/numpy/user/quickstart.html) and in a [Conda](https://docs.conda.io/en/latest/) environment.
If you are not familiar with them, take a few minutes to learn the ins and outs. When you eventually move on to using deep learning for a real robotics task, you will likely be using these tools, as well as [Pytorch](https://pytorch.org/), which is an open source framework for deep learning. PyTorch uses a Numpy-like interface, so it will be good to know the Numpy way of thinking for future assignments and for a healthy, happy life as a machine learning enthusiast, practitioner, or researcher.

## Homework 0. Setup Conda and the codebase ##
First install Miniconda (https://docs.conda.io/en/latest/miniconda.html). Then run the following commands.

```bash
git clone https://github.com/fishbotics/uw-robotics-571-sp21.git
cd uw-robotics-571-sp21
conda deactivate
conda env create -f environment.yml
conda activate robotics-class
```

Although you don't technically _have to_ use Anaconda and Python 3.8.8, this is how we will be grading it, and if it doesn't work on our grading computer, we won't try too hard to make it work.
You should not still be using Python 2 (it's no longer supported, after all) and we have no intention of running your code in a Python 2 environment. However, _if_ you are still using Python 2.7, you can install Anaconda without messing up your current Python installation. Anaconda also makes sure that the installed libraries don't affect other projects on the same machine. There are other tools to do this as well (ask the TAs if you really want to get into it about Python tooling).

Before starting each homework, you will first need to update the repository to get the new files.
```bash
git add -A
git commit -m "before starting new homework"
git pull origin master
```

## [Homework 1. Making Your First Neural Network](hw1)
