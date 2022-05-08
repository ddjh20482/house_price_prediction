
# visualization packages
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Standard data manipulation packages
import pandas as pd
import numpy as np



def heat_map(y_train, X_train):

    heatmap_data = pd.concat([y_train, X_train], axis=1)
    corr = heatmap_data.corr()

    fig, ax = plt.subplots(figsize=(10, 16))
    sns.heatmap(data=corr,
                mask=np.triu(np.ones_like(corr, dtype=bool)),
                ax=ax,
                annot=True,
                cbar_kws={"label": "Correlation", 
                          "orientation": "horizontal", 
                          "pad": .2, 
                          "extend": "both"}
               )
    ax.set_title("Heatmap of Correlation")
    
    plt.show() 
    
    pass

def linearity(y_test, y_pred):

    fig, ax = plt.subplots()

    perfect_line = np.arange(y_test.min(), y_test.max())
    ax.plot(perfect_line, linestyle="--", color="orange", label="Perfect Fit")
    ax.scatter(y_test+3, y_pred, alpha=0.5)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.legend();
    
    pass

import scipy.stats as stats
import statsmodels.api as sm

def qqplot(y_test, y_pred):
    residuals = (y_test - y_pred)
    sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True);
    
def homoscedasticity(y_test, y_pred, X_test):
    residuals = (y_test - y_pred)
    
    fig, ax = plt.subplots()

    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.plot(y_pred, [0 for i in range(len(X_test))])
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Actual - Predicted Value");
    
def price_living_space(est_price):
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(np.arange(1000, 4000, 100), est_price)
    ax.set_title('House Price by Living Space', fontsize=25)
    ax.set_ylabel('House Price', fontsize=18)
    ax.set_xlabel('Squared Feet of Living Space', fontsize=18)
    ax.yaxis.set_major_formatter('${x:1.0f}K')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
    
    pass

def price_floor(est_price):
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh([str(i) for i in range(1, 5)], est_price)
    ax.set_title('House Price by Number of Floors', fontsize=25)
    ax.set_xlabel('House Price', fontsize=18)
    ax.set_ylabel('Number of Floors', fontsize=18)
    ax.xaxis.set_major_formatter('${x:1.0f}K')
    plt.xlim([200, 450])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
    
    pass
    
def est_price_condition(est_price_condition):
    fig, ax = plt.subplots(figsize=(8,4))
    
    condition_name = ['condition_Fair', 'condition_Average', 'condition_Good','condition_Very Good']
    
    ax.barh(condition_name, [i/1000 for i in est_price_condition])
    ax.set_title('House Price by Maintenance', fontsize=25)
    ax.set_xlabel('House Price', fontsize=18)
    ax.set_ylabel('House  Condition', fontsize=18)
    ax.xaxis.set_major_formatter('${x:1.0f}K')
    plt.xlim([200, 450])
    cond_label = ['Fair', 'Average', 'Good', 'Very Good']
    ax.set_yticklabels(cond_label)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()
    
    pass
    
def est_price_reno(est_price_reno):
    
    fig, ax = plt.subplots(figsize=(8,4))
    plt.barh(['NO', 'YES'], [i/1000 for i in est_price_reno])
    ax.set_title('House Price by Renovation', fontsize=25)
    ax.set_xlabel('House Price', fontsize=18)
    ax.set_ylabel('Renovation', fontsize=18)
    ax.xaxis.set_major_formatter('${x:1.0f}K')
    plt.xlim([300, 420])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()

    pass