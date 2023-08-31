# mitigating-multi-type-bias

In the data preprocessing module, datasets for the INLP and CDA methods are constructed based on Wikipedia data. The code for the data preprocessing section is located in the "dataset" folder.

For the model training section, you can locate the implementation code for the INLP debiasing method in "inlp_projection_matrix.py" and for the CDA debiasing method in "run_mlm.py." The INLP debiasing method's implementation is based on the work by Ravfogel et al. (2020), and the CDA debiasing method's implementation is inspired by Zmigrod et al. (2020)

The code for the CrowS-Pairs evaluation method is available in "crows_debias.py," with reference to Nangia et al. (2020).

All experiment execution scripts and results are contained within "Experiments.ipynb."
