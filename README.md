# Income-MLOps
Having fun with OOP and experiment tracking using MLFlow for a simple classification task

Comparing results between XGBoost, scikit-learn and tf NN. Also used NN inside sklearn

Used all default parameters with no hyperparameters search of tuning.

Used 2 sets of data versioning. one is the full dataset and the other is feature selected to compare it.

# Results {model} , {Accuracy} , {Data Version}

1- Xgboost --> "86.2%" , "Full Feature"

2- Random Forest --> "84.8%" , "Full Feature"

3- Xgboost --> "84%" , "Selected Feature"

4- Scikit Neural_Network --> "83.9%", "Full Feature" # I's best to tell that it used 'relu' activation as default parameter

5- Scikit Neural_Network --> "83.4%", "Selected Feature" # I's best to tell that it used 'relu' activation as default parameter
.
.
14- TF Neural_Network --> "79.8%", "Full Feature" # Probably should have picked better architecture. 
