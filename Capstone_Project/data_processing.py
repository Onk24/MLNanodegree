import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def remove_null(df):
    return df.dropna()


def convert_variables(df):
    df["popularity"] = df["popularity"].astype(float)
    df["duration_ms"] = df["duration_ms"].astype(float)
    df["explicit"] = df["explicit"].astype(int)
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["danceability"] = df["danceability"].astype(float)
    df["energy"] = df["energy"].astype(float)
    df["key"] = df["key"].astype(int)
    df["loudness"] = df["loudness"].astype(float)
    df["mode"] = df["mode"].astype(int)
    df["speechiness"] = df["speechiness"].astype(float)
    df["acousticness"] = df["acousticness"].astype(float)
    df["instrumentalness"] = df["instrumentalness"].astype(float)
    df["liveness"] = df["liveness"].astype(float)
    df["valence"] = df["valence"].astype(float)
    df["tempo"] = df["tempo"].astype(float)
    df["time_signature"] = df["time_signature"].astype(int)
    return df


def drop_variables(df):
    return df.drop(["id","name", "artists","id_artists"], axis = 1)


def split_data(df, random_no = 24):
    X = df.loc[:, df.columns != 'popularity']
    y = df.loc[:, 'popularity']
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 1000, random_state = random_no)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 1000, random_state = random_no)
    return X_train.reset_index(drop = True), y_train.reset_index(drop = True), X_val.reset_index(drop = True), y_val.reset_index(drop = True), X_test.reset_index(drop = True), y_test.reset_index(drop = True)


def encode_date(df):
    df["days"] = (datetime.today() - df["release_date"]).dt.days
    return df.drop(["release_date"], axis = 1)

def normalize_data(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_n = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    X_val_n = pd.DataFrame(scaler.fit_transform(X_val), columns = X_train.columns)
    X_test_n = pd.DataFrame(scaler.fit_transform(X_test), columns = X_train.columns)
    return X_train_n, X_val_n, X_test_n


def select_features(X_train, X_val, X_test, features_set):
    return X_train[features_set], X_val[features_set], X_test[features_set]
