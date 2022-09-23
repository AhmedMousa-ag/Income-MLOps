from Utils.config import load_config_file
from Utils.data import load_data
import mlflow
from design_experiment import tf_exper, skl_train, xgb_exp
config_file = load_config_file()
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier # This is interesting because it's Neural Network


tracking_uri = config_file["tracking_uri"]
exp_name = config_file["exp_name"]

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("my-experiment-1")

models = [DecisionTreeClassifier,LogisticRegression,RidgeClassifier, SGDClassifier,
            NearestCentroid,RandomForestClassifier, VotingClassifier, GaussianNB, MLPClassifier]



main_path = config_file["paths"]["prep_data"]


for data_version in config_file["prep_data_names"]:
    x_train, y_train, x_test, y_test = load_data(data_version, main_path)

    
    skl_train(x_train, y_train, x_test, y_test,data_version)
    xgb_exp(x_train, y_train, x_test, y_test,data_version)
    tf_exper(x_train, y_train, x_test, y_test,data_version)