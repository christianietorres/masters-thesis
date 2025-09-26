
# Federated Learning system for Heart Failure classification


This README.md file provides instructions and neccessary information about how to run and navigate this folder to perform Heart failure classification in a federated learning framework. 

opt: formulate better**************************

## The Cleveland dataset 
The dataset used for this classficiation is the Cleveland Heart Disease dataset from the UCI Machine Learning Repository: https://doi.org/10.24432/C52P4X. It consists of 303 samples with 13 features and 1 target feature. The 13 features are age, sex and other heart-related attributes, which makes them vary in data types. The target features is of the integer type and represents how present the heart disease is, with 0 meaning a total absence of the heart disease and the values 1-4 meaning corresponding to the different presence levels of the disease. 

opt:what all the different fetures are*************
source correctly for the dataet**************** 


## The MLP model
The Multiperceptron (MLP) is the neural network model used for training in this federated learning setup. It consists of 1 input layer, 3 hidden layers and 1 output layer, and they respectively have 13, 64, 32, 16 and 1 or 5 nodes. The last node consist of 1 or 5 nodes depends on if the model executes a binary classification or not. 

opt: better last sentence*****************

## How to run the project

### Installation and downloads
1. Download the folder "simple-fl", which has the simple federated learning setup, from the creator at: https://github.com/tie-set/simple-fl

2. Add this folder in the path "simple-fl\examples" to the downloaded folder "simple-fl" from the step before. 

3. Download the Cleveland Heart Disease dataset from: https://doi.org/10.24432/C52P4X. Add that dataset to "simple-fl/examples/HF_classification/data".


4. create a conda environment for the federated learning:

```
# for macOS
conda env create -n federatedenv -f ./setups/federatedenv.yaml

# for Linux
conda env create -n federatedenv -f ./setups/federatedenv_linux.yaml
```

Before running the codes, the environment needs to be activated. You can activate the environment by running the following:

```
conda activate federatedenv
```

opt: write source correctly*****
write step 2 and 3 better + better path*************
opt: better title name*********
is format 1-4 okay?***************



### Execution

Before running everything, make sure the data is processed and distrubuted between clients by running the following:

```pyton
python -m examples.HF_classification.prepare_data.py
```

Run the code below from Ubuntu/Linux terminals. specifically, 4 different Python files need to be run simultaneously from 4 different terminals. Make sure the path is  "\simple-fl" when you run the code files from the terminals. Run the following code snippets in this specific order:


1. For the FL central server 
```python
python -m fl_main.pseudodb.pseudo_db
```

2. For the FL central server 
```python
python -m fl_main.aggregator.server_th
```

3. For the first client agent

```pyton
python -m examples.HF_classification.classification_engine 1 50001 a1
```

4. For the second client agent
```
python -m examples.HF_classification.classification_engine 1 50002 a2
```

ubuntu or linux***********
better path**************
is format 1-4 okay?************
better prestep************

## Additional Instructions
Running this project in a centralized learning setup is optional as well. Do small modifications in the "classification_engine.py"-file, by uncommenting the call to the CL() class and CLRunner() class, and comment out the FL() and FLRunner() class. Rememmer to also run "Prepare_data_CL.py" instead of "Prepare_data.py" to make sure all the data is stored in one place when doing centralized learning. 


two last sentence needs editing**
correct classes?*******

## licene


check all * under the titles

is the format correct?**********
opt: add licene*********
opt: add configuration file?*************
opt: add additional instructions********


