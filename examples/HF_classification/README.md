
# Federated Learning system for Heart Disease classification


This README.md file provides instructions and neccessary information about how to run and navigate this folder to perform heart disease classification in this federated learning (FL) framework. 

It is worth noting that this code **applies and builds on** TieSet's repositiory, which includes the codebase for simplified federated learning [^1]. The dataset used for performing the heart disease classification task is the **Cleveland Heart Disease** dataset from the UCI Machine Learning Repository [^2]. 

- **Location for the unprocessed data (in this repository):** `examples\HF_classification\data\heart+disease`
- **Location for the processed data (output from  prepare_data.py):** `examples\HF_classification\data\clients`

The package `HF_classification` and its files have been added to specifically exectue heart disease prediction. They are heavily inspired by the code file setup and content in Tieset's `image_classification`[^1]. In particular, the file `classification_engine.py` is the main entry point, which runs the whole FL setup. The file is inspired from Tieset's `classification_engine.py`, and `client.py` in `fl_main/agent/client.py` and `fl_main/lib/helpers.py` were modified slightly from the original source [^1].

## How to run the project

### Installation 
1. Clone this repository, and open a terminal at the repo root
```bash
cd /masters-thesis
```

2. Create a conda environment for the federated learning:

```
# for macOS
conda env create -n federatedenv -f ./setups/federatedenv.yaml

# for Linux
conda env create -n federatedenv -f ./setups/federatedenv_linux.yaml
```

3. Before running the codes, the environment needs to be activated. You can activate the environment by running the following:

```
conda activate federatedenv
```


### Data preparation

Before running everything, make sure the data is processed. Based on wheter the system is FL or CL, run **one** of the following:

```pyton
python -m examples.HF_classification.prepare_data.py # for federated data distrubution
```

or 
```pyton
python -m examples.HF_classification.prepare_data_CL.py # for centered data when running the CL setup 
```
### Execution

Make sure to the current path is  '\simple-fl'. Run the code snippets, each in its own terminal, in the following order:


1. Pseudo Database in the FL central server.
```python
python -m fl_main.pseudodb.pseudo_db
```

2. Aggregator in the FL central server .
```python
python -m fl_main.aggregator.server_th
```

3. Client agent 1

```pyton
python -m examples.HF_classification.classification_engine 1 50001 a1
```

4.  Client agent 2
```
python -m examples.HF_classification.classification_engine 1 50002 a2
```

## Optional: centralized learning (CL) setup 
To running this project in a centralized learning (CL) setup, modify `examples/classification/classification_engine.py`-file, by uncommenting the call`to the `CL()` and `CLRunner()` classes`, and commenting out the `FL()` and `FLRunner()` classes.

## Sources
[^1]: https://github.com/tie-set/simple-fl

[^2]: https://archive.ics.uci.edu/dataset/45/heart+disease

## Code Sources
- [tie-set/simple-fl][sfl]

## Other Sources
- [UCI Cleveland Heart Disease dataset][uci]

[sfl]: https://github.com/tie-set/simple-fl
[uci]: https://archive.ics.uci.edu/dataset/45/heart+disease

