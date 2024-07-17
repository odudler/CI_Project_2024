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
- "grid_search.py" and "grokfast.py" contain code for performing hyperparameter tuning

### How to replicate our results:

- Create a conda python environment using python version 3.9:
  - `conda create --name env_name python=3.9`
- Import scikit-surprise via conda, as installing via pip leads to issues:
  - `conda install scikit-surprise`
- Import relevant dependencies using the requirements.txt file:
  - `pip install -r requirements.txt`
- Run the cells in main.ipynb and change the selected model in the third cell to train any implemented model



