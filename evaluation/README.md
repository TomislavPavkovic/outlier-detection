# Evaluation
This package includes a script to compute the dice coefficient, average surface distance, 
hausdorff distance, and 95% hausdorff distance between two NIFTI files.
<br>
It can be used to evaluate the model by comparing model's reconstructions and ground truth images, and to determine the outlier threshold.

## Installation
If outlier-detection environment was installed from the env.yml in the root directory you can skip this env.yml installation.

Create a new conda environment by **navigating to this directory** and executing the following command:
```shell
conda env create --name evaluation --file=env.yaml
```

Now, activate the new environment:
```shell
conda activate evaluation
```

Then, install this project in the editable mode by **navigating to the project's root directory** and executing the
following command:
```shell
conda develop .
```

## Configuration
All changeable parameters of this part of the project can be configured via the `config.yaml` file.
Please have a look at the comments in the config file for more information.

## Execution
If the configuration is done, run `evaluate.py` without command line arguments.
