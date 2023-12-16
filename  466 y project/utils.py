import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

split_ratio = 0.4

attribute = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
point = 'quality'

def getData(wine_type):
    if wine_type == "red":
        dataset = pd.read_csv("winequality-red.csv")
        dataset = dataset.sample(frac=1)
        
        X_all = dataset.drop("quality", axis=1)
        y_all = dataset['quality']
         
        X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    elif wine_type == "white":
        dataset = pd.read_csv("winequality-white.csv")
        dataset = dataset.sample(frac=1)
        X_all = dataset[attribute].values.reshape(-1, len(attritube))
        y_all = dataset[point].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def accuracy(pred, y_test):
    correct = sum(1 for p, y in zip(pred, y_test) if p == y)
    acc = correct / len(pred)
    return acc

def close_accuracy(pred, y_test):
    close = sum(1 for p, y in zip(pred, y_test) if p == y or p + 1 == y or p - 1 == y)
    acc = close / len(pred)
    return acc
