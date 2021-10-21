# CPSC8420Project Phase 2
This code give an add-on to Bayesian Algorithm Execution (BAX) presented in the paper [Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information](https://arxiv.org/abs/2104.09460)
```
@inproceedings{neiswanger2021bayesian,
  title         = {Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information},
  author        = {Neiswanger, Willie and Wang, Ke Alexander and Ermon, Stefano},
  booktitle     = {International Conference on Machine Learning},
  year          = {2021},
  organization  = {PMLR}
}
```

Here an example of multi-objective optimization is presented with function evaluations existing in 2 dimensions. 

# Installation

Python 3.6+ is required. Run the following commands to install requirements and dependencies
```
pip install -r requirements/requirements.txt

pip install -r requirements/requirements_gpfs.txt
```

Before examples can be ran, the following two commands must be ran
```
python bayesian-algorithm-execution/bax/models/stan/compile_models.py

source bayesian-algorithm-execution/shell/add_pwd_to_pythonpath.sh
```

# Run Example
To run the multi-objective optimization example, run the following command
```
python nonDominated.py
```
The current number of iterations is set to 20, but can be changed by adjusting the n_iter variable in the file. The plots for each iteration will be saved in nonDom_pareto. 

# Recommendation
It is recommended to use an anaconda virtual enviornment. This is not required but can make the installation process easier. Install anaconda, and do the following steps before installing the dependancies
```
conda create --name myenv python=3.6

conda activate myenv
 ```
