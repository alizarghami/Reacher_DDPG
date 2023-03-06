
# Solving Reacher problem from Unity Environments using Deep Reinforcement Learning

### Introduction

This repository contains code for training an intelligent agent that can act efficiently in Reacher unity environment using Deep Reinforcement Learning.

### About the environment

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The goal is to reach score of +30 over 100 consecutive episodes.

### Getting Started

In order to run this code you need to have Python 3 and Jupyter-notebook installed. In addition you need to install the following modules.
* Pytorch: [click here](https://pytorch.org/get-started/locally)
* Numpy: [click here](https://numpy.org/install)
* UnityEnvironment: [click here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
* OpenAI Gym: [click here](https://github.com/openai/gym)

You also need to download the Reacher environment from the links below. You need only select the environment that matches your operating system:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
Make sure to decompress the zipped file before running the code

You are strongly suggested to install all the dependencies in a virtual environment. If you are using conda you can create and activate a virtual environment by the following commands:

	conda create --name ENVIRONMENT_NAME python=3.6
	conda activate ENVIRONMENT_NAME
	
You can deactivate your environment by this command:

	conda deactivate
	
An alternative method for using python virtual environments can be found here: [click here](https://virtualenv.pypa.io/en/latest/)

For more information and instructions on how to install all dependencies check [this link](https://github.com/udacity/deep-reinforcement-learning#dependencies).

### Instructions

In order to run the code you need to open `reacher.ipynb` in your Jupyter-notebook. Point to the Navigation Environment location on your system where specified in the code and run.
