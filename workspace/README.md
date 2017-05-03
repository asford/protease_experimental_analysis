# Overview

* `protease_experimental_analysis.protease_sequencing_model` - Analysis of protease susceptibility screening.
* `protease_experimental_analysis.sequence_protease_susceptiblity` - Sequence-based protease susceptibility modeling.
* `protease_experimental_analysis.data` - Contains all raw counts data in the *.counts files, as well as sorting conditions in experiments.csv
* `workspace` - Model analysis and development workspace.


# Quickstart

This module may be (A) added directly to the python search path or (B) installed as a module.

## Direct Use
1) `pip install -r requirements.txt` to install module requirements.

2) ``PYTHONPATH=`pwd` jupyter notebook`` to launch a notebook server with `protease_experimental_analysis` available.

## Module

1) `pip install -e .` to perform an editable/development install of the module. Use of a virtualenv strongly recommended.

## To re-fit all EC50 values from the raw sequencing data:

From inside the workspace directory:
`python fit_all_ec50_data.py`

## To re-fit the unfolded state model:

From inside the workspace directory:
`python fit_unfolded_state_model.py`
