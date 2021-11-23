# CPSC8420Project Phase 3
This code goes along with the paper titled Applying Logarithmic Time Gaussian Process Regression To BAX by Madison Krell, makrell@clemson.edu. 

This code apply the process from https://github.com/EEA-sensors/parallel-gps for the paper https://arxiv.org/abs/2102.09964
## Citation
```
@article{corenflos2021,
  title={Temporal {G}aussian Process Regression in Logarithmic Time},
  author={Adrien Corenflos and Zheng Zhao and Simo S\"{a}rkk\"{a}},
  journal={arXiv preprint arXiv:2102.09964},
  year={2021}
}
```

to Bayesian Algorithm Execution (BAX) presented in the paper [Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information](https://arxiv.org/abs/2104.09460)
```
@inproceedings{neiswanger2021bayesian,
  title         = {Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information},
  author        = {Neiswanger, Willie and Wang, Ke Alexander and Ermon, Stefano},
  booktitle     = {International Conference on Machine Learning},
  year          = {2021},
  organization  = {PMLR}
}
```


# Installation

Python 3.6/3.7 is required. To make use of the speedup and parellization, the system must have a GPU.Run the following commands to install requirements and dependencies
```
pip install -r requirements/requirements.txt

pip install -r requirements/requirements_gpfs.txt
```

Before examples can be ran, the following two commands must be ran
```
python bayesian_algorithm_execution/bax/models/stan/compile_models.py

source shell/add_pwd_to_pythonpath.sh
```

# Run Example
To run the example uisng the pssgp method, run the following command
```
python parallel_test.py.py
```
The current number of iterations is set to 20, but can be changed by adjusting the n_iter variable in the file. The plots for each iteration will be saved in parallel_topk_results. 

# Recommendation
It is recommended to use an anaconda virtual enviornment. This is not required but can make the installation process easier. Install anaconda, and do the following steps before installing the dependancies
```
conda create --name myenv tensorflow-gpu python=3.7

conda activate myenv
 ```
