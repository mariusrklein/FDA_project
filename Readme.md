# Correction of ion suppression in single cell metabolomics datasets

This project investigates whether an additional correction step in the single-cell metabolomics pipeline SpaceM improves the accuracy of cellular metabolic profiles. To this end, pre-existing R code is being implemented in python and tested on multiple metabolomics and lipidomics datasets. As readout, the separation of cellular populations is examined visually and quantitatively.

This work does not try to tackle ion suppression itself, but rather equalize its effect on all ablation marks of a sample. This is achieved by regressing out the dependency of intensities on the sampling proportion of an ablation mark.


## Installation

It is advised to install this tool inside a virtual environment to prevent conflicts with other installed packages (especially SpaceM).
To create a virtual environment, execute the following commands:

```
$ python3 -m venv /path/to/your/virtual_environment/
$ source /path/to/your/virtual_environment/bin/activate
```

The Python command line application can be downloaded from this Github repo via pip. Just run this command in your Terminal:
```
$ pip install git+https://github.com/mariusrklein/FDA_project.git@package
```

It installs the package and all dependencies (including a forked version of SpaceM from git.embl.de, requires EMBL credentials).

## Usage

The correction can be applied to any SpaceM-generated data folder that contains individual wells as subfolders. It should also contain a metadata file that gives an overview of the wells to be included in correction and evaluation.

```
dataset_folder/
├─ metadata.csv
├─ well_X/
│  ├─ analysis/
│  │  ├─ ablation_mark_analysis/
│  │  │  ├─ spatiomolecular_adata.h5ad
│  │  ├─ overlap_analysis2/
│  │  │  ├─ ablation_mark.regions.csv
│  │  │  ├─ cell.regions.csv
│  │  │  ├─ overlap.regions.csv
│  │  ├─ single_cell_analysis/
│  │  │  ├─ spatiomolecular_adata.h5ad
│  ├─ config.json
├─ .../
```

Folders and files can be named differently, this is defined in the correction_config.json file which can be generated using the terminal command:

```
$ python3 -m scmIonSuppressionCorrection /path/to/dataset/ --make-config
```
This configuration file also defines all aspects of the correction, deconvolution and evaluation, like 

 - 'correction_proportion_threshold': the minimum cellular sampling proportion of pixels to be included in quantile regression for correction
 - whether the deconvolution parameters from SpaceM's well-specific config.json files should be used. If not, deconvolution parameters can be specified manually.
 - whether quality control and evaluation notebooks should be executed
 - where corrected data matrices should be saved

 The standard configuration is displayed and roughly explained [here](https://github.com/mariusrklein/FDA_project/blob/package/scmIonSuppressionCorrection/src/const.py).

To run the ion suppression correction workflow, the same command is run without the `--make-config` flag.
Instead, other flags can be used to specify a correction_config file (-c), the number of cores to use in computation (-j) or whether to run in verbose mode (-v).
Thus, a typical workflow command would look like this:

```
$ python3 -m scmIonSuppressionCorrection /path/to/dataset/ -c /path/to/config/correction_config.json -j48 -v
```



