# CI_Project_2024
## Project from Computational Intelligence Lab ETHZ 2024

### Extending Neural Collaborative Filtering Algorithms and comparing performance to 5 baseline algorithms:

- k-NN
- Bayesian Factorization Machine (BFM)
- SVD
- SVD++
- Neural Collaborative Filtering (NCF)


### Project Code is structured as follows:

- "data" folder contains training data and a submission template file
- "ncf" folder contains implementation of the ncf algorithm + extensions
- "outputs" contains submission csv files containing the predicted ratings
- "main.ipynb" is a notebook for running gridsearch and creating submissions
- "models.py" contains implementations of kNN, SVD, SVD++ and BFM algorithms
- "grid_search.py" contain code for performing hyperparameter tuning
- "grokfast.py" contains code copied from the Grokfast repo which implements the Grokfast algorithm

### How to replicate our results:

#### Setup Environment

- Create a conda python environment using python version 3.9:
  - `conda create --name env_name python=3.9`
- Import scikit-surprise via conda, as installing via pip leads to issues:
  - `conda install scikit-surprise`
- Import relevant dependencies using the requirements.txt file:
  - `pip install -r requirements.txt`
 
#### Replicate Submission Files for baseline algorithms except NCF
In order to replicate the prediction files for submission in kaggle do the following:
Run the cells in main.ipynb and change the selected model in the third cell. The third cell contains the following line of code: 
`create_submission('BFM', 'outputs/submission_file.csv')`
Changing the first string parameter to any of ['SVDplusplus', 'SVDsimple', 'BFM', 'KNN'] uses the corresponding model for training. The second parameter denotes the path and name of the output .csv file.

#### Replicate Submission Files for NCF algorithm + extended version

TODO

#### Replicate hyperparameter tuning results for baseline algorithms except NCF

TODO

#### Replicate hyperparameter tuning results for NCF algorithm + extended version

TODO





