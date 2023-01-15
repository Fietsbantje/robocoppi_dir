## The back end

This repo provides the **Python** code for building the back end of a machine learning chat bot. 
The machine learning model of this bot was built using **Tensorflow**, an open-source platform for machine learning:  

https://www.tensorflow.org/overview

It also uses **nltk**, a platform for building Python programs working with human language data.

https://www.nltk.org/index.html

## Credits and links  

For this first repository, robocoppi_dir, I followed this tutorial:  

https://www.techwithtim.net/tutorials/ai-chatbot/  

I made some changes to get it to work, as this tutorial is a bit dated by the time I am writing this:

I used VS Code as a code editor and created a Conda environment to solve some issues with the Python packages. 
I changed all the variable names to more explanatory names, in accordance with the current terminology within the domain of Conversation Design.  

The tutorial wasn't designed for beginners and having knowledge of Python is essential to really understand this code. 
I followed a few open-source tutorials to learn Python, created by the same teacher (Tim Ruscica aka Tech with Tim) and eventually his ProgrammingExpert Course at AlgoExpert.  

## RoboCoppi artwork  

The GitHub persona of RoboCoppi was designed by Stellar (Steve Napier):  
https://instagram.com/stellar_steve_napier?igshid=YmMyMTA2M2Y=  

## Initial Setup 

Clone repo  
```
$ git clone https://github.com/Fietsbantje/robocoppi_dir.git  
$ cd robocoppi_dir
```
Use **Conda** to create an environment and then activate it.  
In the following example the name of the environment is coppi-env and the Python version used is 3.10  
```
$ conda create --name coppi-env python=3.10  
$ conda activate coppi-env  
```
Install packages: 
You can use **Pip** in a Conda environment. Pip is included in Anaconda and Miniconda. 
Note that you can't use 'conda install' after 'pip install'. First install what you want to install with Conda. After that install packages that can't be installed with Conda, using pip.  
```
$ (coppi-env) conda install -c anaconda numpy nltk tflearn
$ (coppi-env) pip install tensorflow
```
Import nltk package and download punkt.
Note that you type this in the **Python Interpreter** (the $ in the prompt changes in >>>). 
Typing python (or python3 on Linux) in the Command Line will activate the Python interpreter and you can get back to the Command Line prompt typing CTRL + d
```
$ (coppi-env) python  
>>> import nltk  
>>> nltk.download('punkt')  
```
Modify intents.json with different intents and responses for your Chatbot  

Run 
```
$ (venv) python robocoppi.py  
```
This will dump model.tflearn.data file. 


