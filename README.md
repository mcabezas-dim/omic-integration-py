# Omic Toolbox for Python

This project contains python scripts for data analysis of omic data. The main tasks that one can do with this are:
* preprocessing data and metadata for omic (drop missing values, label standarization),
* integration and normalization of data,
* reduction of dimensionality for statistical analysis,
* clusterization of data,
* Pareto Task Inference for metadata enrichment,
* prediction models for feature importance,
* utilities like plots and tables creation.  

This toolbox is intended for "simple" analysis, and designed to run locally in a every-day computer. Despite this, one can adapt the source code for other purposses.  
## Instalation
Current version uses python 3.14.

To install and use the toolbox, we suggest to follow the instructions below. 
1. install `Anaconda` for a more interactive and beginner-friendly interface, or `Miniconda` for a more simple and compact use.
2. download this repository in your device.
3. install the environment that comes in this repository (called `environment.yml` file). This is usually done by running the following command in the Anaconda Prompt:
      ```conda env create -f environment.yml```.
Be sure to do this in the same directory where you downloaded this repository. 

After the CONDA environment is created, one is ready to use all the tools in this project. 

For a tutorial on how to use this tools, activate the environment, run Jupyter Lab (or similar), and open the `tutorial.ipynb` notebook.

For any questions about the tutorial, feel free to contact.
