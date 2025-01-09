9.1.2025

Be aware of the slashes (/ or \) in the filepaths according to the system used to run these codes.

WORKFLOW:
1. Process the raw UE signal features, this is done in 2 parts (2 matlab files)
	1.1 mlp_data_generation.m (in the ROOTFOLDER/mlp_data_generation folder). This code preprocesses the raw features --> MLP_input_cov.mat or MLP_input_log.mat (logarithmic). It is unnecessary to to do this step again.
	1.2 biggerData_*.m files
2. Python: goodConfigGPU*.py files. This step trains the ML model and runs the testing (online) dataset through the model in order to get the z values to be used later in the matlab code.
3. Matlab: everything*.m files. This step will produce the final results of the localization/distance prediction, and compare both metric learning and euclidean distance approaches.

Please go through all these steps (apart from 1.1) to evaluate the results. This is because some of the data files have similar names and in some cases shuffling of the date is involved, so not following the steps might lead to false results. 

DOCUMENTATION:
MATLAB CODE:
WKNN:
The code files have WKNN in their name, everything_WKNN_grid_real_comp.m

PAIRWISE DISTANCE PREDICTION:
The code files have _d_ in their name, e.g. biggerData_d_grid_real.m

NAMING OF THE FILES:
biggeData._ files create testing (online) and training (offline) datasets for the Pytorch ML models
everything._ files are used to assess either the localization accuracy (WKNN), or the pairwise distance prediction (d), they use the outputs of the pytorch models (and also the biggerData outputs).

_real_ means that none of the online data is seen in the training --> realistic
_grid_ means that the training samples are organized in a grid structure
_cov_ means that rather than using log covariance as the signal feature the plain covariance is used (bad results)
_k_ means that only k closest neighbors in terms of euclidean feature distance were used in the training of the model (to make the model more sensitive to shorter distances)
_comp_ means comparison of using models trained with all neighbors and models trained with k closest neighbors
_all_ means that the WKNN script goes over all k values that have been used to train the ML model

grid_tester.m is used to calculate the Pearson corr coefficients of the data used in WKNN


PYTHON:
The naming of the files follows same principles in the Matlab code.

The python scripts are meant to be run with GPUs due to the high number of epochs.
Also, I used the Triton resources, so I ran the python codes with GPUscript.sh, and the output is printed in testGPU.out

DATA FILES:
Python outputs (z values) are in the outputs folder.
Other data is in the root folder.

Similar naming logic applies