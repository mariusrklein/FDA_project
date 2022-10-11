# Correction of ion suppression in single cell metabolomics datasets

This project investigates whether an additional correction step in the single-cell metabolomics pipeline SpaceM improves the accuracy of cellular metabolic profiles. To this end, pre-existing R code is being implemented in python and tested on multiple metabolomics and lipidomics datasets. As readout, we measure the performance a simple LDA classifier to discriminate between cell types or conditions from a dataset.

This work does not try to tackle ion suppression itself, but rather equalize its effect on all ablation marks of a sample. This is achieved by regressing out the dependency of intensities on the sampling proportion of an ablation mark.