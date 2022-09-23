from statistics import mode
import tensorflow as tf
from tensorflow.keras.layers import Dense
from Utils.mlflow import mlflow_track
import mlflow
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


@mlflow_track
def xgb_exp(x_train, y_train, x_test, y_test, data_version):
    print("Starting XGBoost Classifier Experiement")
    mlflow.xgboost.autolog()
    mlflow.log_param("Data Version", data_version)
    model = XGBClassifier()
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1_score)


@mlflow_track
def skl_train(x_train, y_train, x_test, y_test, data_version, model):
    mlflow.sklearn.autolog()
    mlflow.log_param("Data Version", data_version)
    model = model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1_score)


def skl_exp(x_train, y_train, x_test, y_test, data_version, models):
    print("Starting Sklearn Experiement")
    model_num = 0
    for model in models:
        model_num += 1
        print(model_num)
        skl_train(x_train, y_train, x_test, y_test, data_version, model)


def build_model(num_layers=3, nuerons_num=30):
    model = tf.keras.Sequential()
    for _ in range(num_layers - 1):
        model.add(Dense(nuerons_num))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics="accuracy")
    return model


@mlflow_track
def tf_exper(x_train, y_train, x_test, y_test, data_version):
    print("Starting Tensorflow Experiement")
    mlflow.tensorflow.autolog()
    mlflow.log_param("Data Version", data_version)
    model = build_model()
    model.fit(x_train, y_train)
    loss, acc = model.evaluate(x_test, y_test)
    mlflow.log_metric("accuracy", acc)
    # mlflow.log_metric("f1_score",f1_score) # Tensorflow doesn't support f1_score, so we have to write it if we need it, inherit from keras.metrics
