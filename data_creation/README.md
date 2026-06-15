# Data Creation #

This folder consists of scripts related to generating Spin-1 and Spin-1/2 sample data. The primary script to run is `Create_Training_Data.py`, in which you can edit CLI arguments to adjust the MC generator's configurations.

`Create_Data.slurm` is a template SLURM script for running massively parallel jobs to create a large amount of training data within a very large amount of data (operating time is $\\~O(n^2)$ per script, approximately). Two other scripts, `merge.py` and `run_signal_generator.py`. Once all jobs submitted via the single SLURM script are finished, `merge.py` will be submitted via `merge.sh` from within `Create_Data.slurm` to merge all of the sample data into a single file. 