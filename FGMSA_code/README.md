## FGMSA

### 1. Get Started

`test_feature.pkl` is a feature file containing only test set.

`models_trained` is a folder containing trained models.

Use command line in this folder:

  ```bash
  # show usage
  $ python run_test.py -h

  # test trained model
  # trained models is always in "models_trained" folder
  $ python run_test.py ./models_trained/fg-mlf-dnn-acc2.pth
  ```

### 2. Dependency Installation

Normally the pip install command will handle the python dependency packages. The required packages are listed below for clarity:

- python >= 3.8
- torch >= 1.9.1
- transformers >= 4.4.0
- numpy >= 1.20.3
- pandas >= 1.2.5
- tqdm >= 4.62.2
- scikit-learn >= 0.24.2
- easydict >= 1.9
- The feature file is large (about 1.6G) and can be downloaded from https://1drv.ms/u/s!AjnIA7eR-eoWhTvAGRK3kmrpaDJ6?e=lRhVgP. To run the souce code, the feature file should be placed in the folder FGMSA_code.
