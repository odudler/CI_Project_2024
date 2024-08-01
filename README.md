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
- "grokfast.py" contains code copied from the [Grokfast repo](https://github.com/ironjr/grokfast) which implements the Grokfast algorithm

### How to replicate our results:

#### Setup Environment

- Create a conda python environment using python version 3.9:
  - `conda create --name {env_name} python=3.9`
- To activate the environment, run:
  - `conda activate {env_name}`
- Import scikit-surprise via conda, as installing via pip leads to issues:
  - `conda install conda-forge::scikit-surprise==1.1.3`
- Import relevant dependencies using the requirements.txt file:
  - `pip install -r requirements.txt`
 
#### Replicate Submission Files for baseline algorithms except NCF
In order to replicate the prediction files for submission in kaggle do the following:
Run the cells in main.ipynb and change the selected model in the third cell. The third cell contains the following line of code: 
`create_submission('BFM', 'outputs/submission_file.csv')`
Changing the first string parameter to any of ['SVDplusplus', 'SVDsimple', 'BFM', 'KNN'] uses the corresponding model for training. The second parameter denotes the path and name of the output .csv file.

#### Replicate Submission Files for NCF algorithm + extended version

Run `python ncf/ncf_train.py --model {model}`, where `{model}` is in:

- `ncf`: Base Neural Collaborative Filtering
- `ncf_extended`: Extended Neural Collaborative Filtering (Our best model)
- `ncf_extended_attention`: Extended Neural Collaborative Filtering with attention
- `ncf_extended_same`: Extended Neural Collaborative Filtering including User-User and Item-Item similarities in the GMF model
- `ncf_extended_cross`: Extended Neural Collaborative Filtering including User-Item and Item-User similarities in the GMF model
- `ncf_extended_same_cross`: Extended Neural Collaborative Filtering including User-User, Item-Item, User-Item and Item-User similarities in the GMF model
- `ncf_extended_gmf`: Standalone GMF model of Extended Neural Collaborative Filtering
- `ncf_extended_mlp`: Standalone MLP model of Extended Neural Collaborative Filtering
- `ncf_extended_gmf_int_default`: Extended Neural Collaborative Filtering with 'average' aggregation method in the GMF model
- `ncf_extended_gmf_int_weight`: Extended Neural Collaborative Filtering with 'weight by rating' aggregation method in the GMF model
- `ncf_extended_mlp_int_weight`: Extended Neural Collaborative Filtering with 'weight by rating' aggregation method in the MLP model
- `ncf_extended_mlp_int_split`: Extended Neural Collaborative Filtering with 'split by rating' aggregation method in the MLP model
- `ncf_extended_no_int`: Extended Neural Collaborative Filtering without any aggregated embeddings

to run training for these models. Each model will be trained five times on five different train-validation splits with 80% training data. The script will print the mean RMSE and the standard deviation of the RMSE over these five runs. We take the model that achieved the lowest validation RMSE during training for each of the five runs as the final model of that run. The code will also generate submission files for the best model each run. The first three models are used for the results in the main paper, while the rest are used for the more detailed experiments in the appendix.

#### Replicate hyperparameter tuning results for baseline algorithms except NCF

Execute the cells in the notebook `grid_search.ipynb`. Each parameter search will train the model on four-fifths of the total data and test it on one-fifth of the data to determine the optimal parameters. The searches will be parallelized across all available cores, and the scores for each parameter set will be printed.

#### Replicate hyperparameter tuning results for NCF algorithm + extended version

Run `python ncf/ncf_hyperparam.py --search_space {model}`, where `{model}` is either 'ncf' or 'ncf_extended'. The script will perform the hyperparameter search that we performed to find the best hyperparameters for the NCF and Extended NCF models.





