from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# you can train the logistic regression from the patch or from mnist
mode = 'patch' # mnist or patch

if mode == 'mnist':
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='auto')
    dim_row = 28
    dim_col = 28
    iters = 30
else:
    # change file location to correct file location
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/all_drawings.json", 'r') as file:
        data = json.load(file)
    X = np.array([np.ravel(d[0]) for d in data])
    y = np.array([d[1] for d in data])
    dim_row = 27
    dim_col = 19
    iters = 1000 


def plot_digits(X_test, y_test):
    plt.figure(figsize=(20,6))
    for i in range(10):
        if np.where(y_test==f"{i}")[0].size > 0:
            index = np.where(y_test==f"{i}")[0][0]
            digit_sub = plt.subplot(2, 5, i + 1)
            digit_sub.imshow(np.reshape(X_test[index], (dim_row,dim_col)), cmap=plt.cm.gray)
            digit_sub.set_xlabel(f"Digit {y_test[index]}")
    plt.show()

def plot_weights(coef):
    
    scale = np.abs(coef).max()
    plt.figure(figsize=(10,5))
    plt.title("Coefficients for model")    
    for i in range(10): # 0-9
        coef_plot = plt.subplot(2, 5, i + 1)

        coef_plot.imshow(coef[i].reshape(dim_row, dim_col), 
                        cmap="seismic",
                        vmin=-scale, vmax=scale,
                        interpolation='bilinear')
        
        coef_plot.set_xticks(()); coef_plot.set_yticks(())
        coef_plot.set_xlabel(f'Class {i}')

    plt.show()