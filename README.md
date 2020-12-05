# Histopathology

CNNs to Detect Metastatic Cancer in Pathology Slides. 

## About

### Repository Structure

* `notebooks/`: An assortment of experiments on data loading, visualization, model structure, and more.
* `noteboks/lib/`: Finalized implementations of models, datasets, transforms, and more.
* `figures/`: Graphics generated during exploration, experimentation, and evaluation.

### Data Download

It is recommended to download the dataset (~7.4G) from [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/) and store it in the root of this repository as `data/`.

Alternatively, data can be downloaded onto Google Drive, and accessed by running the
notebooks on Google Colab. See `notebooks/download_data.ipynb` for details.

### Stack

The experiments in this repository are built using `PyTorch` and `Torchvision`.

## References

1. B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". arXiv:1806.03962