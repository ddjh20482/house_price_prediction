import pandas as pd
import seaborn as sns
import numpy as np

# handling missing values
def missing(house):
    
    # "waterfront" column
    house.waterfront = house.waterfront.fillna('NO')

    # "house" column
    house.view = house.view.fillna('NONE')

    # "is_renovated" column
    # create a new column and set a default value to be 'NO'
    # set the value to be 'YES' if a proper year value exists
    # delete the old column
    house['is_renovated'] = 'NO'
    house.loc[house['yr_renovated'] > 0, 'is_renovated'] = 'YES'
    house.drop(['yr_renovated'], axis = 1, inplace=True)
    
    return house

# cleaning data
def cleaning(house):
    
    # "view" column
    house.loc[house['view'] != 'NONE', 'view'] = 'YES'

    # "condition" column
    house.loc[house['condition'] == 'Poor', 'condition'] = 'Fair'

    # "grade" column
    house.loc[house['grade'] == '4 Low', 'grade'] = '6 Low Average'
    house.loc[house['grade'] == '5 Fair', 'grade'] = '6 Low Average'
    house.loc[house['grade'] == '3 Poor', 'grade'] = '6 Low Average'
    house.loc[house['grade'] == '12 Luxury', 'grade'] = '11 Excellent'
    house.loc[house['grade'] == '13 Mansion', 'grade'] = '11 Excellent'

    # "sqft_basement" column
    # handling missing values first to have consistant data type
    house.loc[house['sqft_basement'] == '?', 'sqft_basement'] = 0.0

    # set a default value for a new column
    house['has_basement'] = 'NO'

    # assign a new value
    house.loc[house['sqft_basement'].astype('float64') > 0, 
              'has_basement'] = 'YES'

    house.drop(['sqft_basement'], axis = 1, inplace=True)
    
    return house

# normalization using mean and standard deviation
def normalize(x):
    return (x - np.mean(x))/np.std(x)

# log transformation and normalization
def numeric_transform(house):
    
    # select columns with numeric data type
    house_numeric = house.select_dtypes(include = ['float64', 'int64'])

    # log transformation
    house_num_log = pd.DataFrame([])
    for col in house_numeric.columns:
        house_num_log[col] = house_numeric[col].map(lambda x: np.log(x))

    # return normalized columns
    return house_num_log.apply(normalize)

# log transformation
def log_transform(house):

    # select columns with numeric data type
    house_numeric = house.select_dtypes(include = ['float64', 'int64'])

    # log transformation
    house_num_log = pd.DataFrame([])
    for col in house_numeric.columns:
        house_num_log[col] = house_numeric[col].map(lambda x: np.log(x))

    return house_num_log

from sklearn.preprocessing import OrdinalEncoder

def categorical_tansformation(house):
    
    # function for encoding binary columns as 1's and 0's
    def binary_encoder(col):
        column = house[[col]]
        encorder = OrdinalEncoder()
        encorder.fit(column)
        encoded = encorder.transform(column)
        return encoded

    # four columns are transformed using binary_endocer
    house['waterfront'] = binary_encoder('waterfront')
    house['is_renovated'] = binary_encoder('is_renovated')
    house['has_basement'] = binary_encoder('has_basement')
    house['view'] = binary_encoder('view')

    # each of two columns are transformed into multiple binary columns
    # the first column of each of two columns is removed
    multi_cat = ['condition', 'grade']
    dummies = pd.get_dummies(house[multi_cat], 
                             prefix = multi_cat, 
                             drop_first=True)
    house = house.drop(multi_cat, axis=1)
    return pd.concat([house, dummies], axis = 1)


def concatenation(house_num_final, house):
    # selecting all binary columns saved into the original dataset
    cat = ['waterfront', 'view','is_renovated', 'has_basement',
           'condition_Fair', 'condition_Good', 'condition_Very Good',
           'grade_11 Excellent', 'grade_6 Low Average', 'grade_7 Average',
           'grade_8 Good', 'grade_9 Better']
    
    return pd.concat([house_num_final, house[cat]], axis = 1)

#Linear Regression definition
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# setting default split values
from sklearn.model_selection import cross_validate, ShuffleSplit
splitter = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def scores(X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    baseline_scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        return_train_score=True,
        cv=splitter
    )
    print("Train score:     ", baseline_scores["train_score"].mean())
    print("Validation score:", baseline_scores["test_score"].mean())

    print("X-test score:    ", model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    print("R2 score:        ", r2_score(y_test, y_pred))

    print("Mean**2 Error:   ", mean_squared_error(y_test, y_pred, squared=False))
    
    pass

from statsmodels.stats.outliers_influence import variance_inflation_factor

# return scores of multicollinearity of each variable
def multicollinearity(X_train):

    vif = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    
    return pd.Series(vif, index=X_train.columns, name="Variance Inflation Factor")

# normalization
def norm(x, col, house_num_log):
    return (x - np.mean(house_num_log[col]))/np.std(house_num_log[col])

# inverse normalization
def denorm(x, col, house_num_log):
    return np.exp(x * np.std(house_num_log[col]) + np.mean(house_num_log[col]))

# House price point estimation
def estimation(est, house_num_log):
    est_dummy = est.copy()
    
    coef = [ 0.33133604, -0.06786479,  0.08805675, -0.27924776,  
            0.97882272, 0.26035277,  0.10187172,  0.13554265, 
            -0.30771804,  0.04835159, 0.19574832, .06361813,
            0.53238091, -1.85648387, -1.34676712, -0.89819289, -0.40143734, 3.97050031,
            0.04492393]
    
    keys = [i for i in est_dummy.keys()]
    
    for i in range(4):
        est_dummy[keys[i]] = norm(np.log(est_dummy[keys[i]]), keys[i], house_num_log)

    est_dummy['interaction'] = est_dummy['sqft_living'] * est_dummy['yr_built']

    keys = [i for i in est_dummy.keys()]
    
    log_est_price = model.intercept_

    for i in range(len(keys)):
        log_est_price += est_dummy[keys[i]] * coef[i]

    est_price = denorm(log_est_price, 'price', house_num_log)

    return est_price

# House price estimation of a range of values after selecting a column
def estimation_with_col(est, val, col, house_num_log):
    est_dummy = est.copy()
    est_dummy[col] = val

    coef = [ 0.33133604, -0.06786479,  0.08805675, -0.27924776,  
            0.97882272, 0.26035277,  0.10187172,  0.13554265, 
            -0.30771804,  0.04835159, 0.19574832, .06361813,
            0.53238091, -1.85648387, -1.34676712, -0.89819289, -0.40143734, 3.97050031,
            0.04492393]
    
    keys = [i for i in est_dummy.keys()]
    
    for i in range(4):
        est_dummy[keys[i]] = norm(np.log(est_dummy[keys[i]]), keys[i], house_num_log)

    est_dummy['interaction'] = est_dummy['sqft_living'] * est_dummy['yr_built']

    keys = [i for i in est_dummy.keys()]

    log_est_price = model.intercept_

    for i in range(len(keys)):
        log_est_price += est_dummy[keys[i]] * coef[i]

    est_price = denorm(log_est_price, 'price', house_num_log)

    return est_price

# estimate price by conditions
def est_price_condition(est, house_num_log):

    est_price_condition = []
    condition_name = ['condition_Fair', 'condition_Average', 'condition_Good','condition_Very Good']
    for name in condition_name:
        est_condition = est.copy()
        for n in condition_name:
            est_condition[n] = 0
        est_price_condition.append(estimation_with_col(est_condition, 1, name, house_num_log))

    return est_price_condition

# estimate price by renovation condition
def est_price_reno(est, house_num_log):

    est_price = []
    col_name = 'is_renovated'
    for i in range(2):
        est_binary = est.copy()

        est_price.append(estimation_with_col(est_binary, i, col_name, house_num_log))

    return est_price

from itertools import combinations

from sklearn.model_selection import cross_val_score, KFold

def interaction(X_train, y_train):
    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    regression = LinearRegression()
    
    interact = []
    x1 = []
    x2 = []
    X_interact = X_train.copy()
    for comb in combinations(X_train.columns, 2):

        x_1, x_2 = comb

        X_interact['interaction'] = X_interact[x_1] * X_interact[x_2]

        interact.append(np.mean(cross_val_score(regression, 
                                                X_interact, 
                                                y_train, 
                                                scoring='r2', cv=crossvalidation)))
        x1.append(x_1)
        x2.append(x_2)

    df_interact = pd.DataFrame({
        'interact': interact,
        'x1': x1,
        'x2': x2
    })

    return df_interact.sort_values(by='interact', ascending=False)