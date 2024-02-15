import numpy as np
import pandas as pd
import pandas_datareader.data as web
import time
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
from bs4 import BeautifulSoup

import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix

############################################################################################################################################
#Section 1: Data Sourcing

############################################################################################################################################
#Section 2: Feature Selection
def get_factor_stats(factors, response):
    """
    Input:
        factors: pd.DataFrame. factors data without response variable.
        response: pd.DataFrame. Response variable data.
    Output:
        df_stat: pd.DataFrame. Statistics table for factors.
    """
    
    stats = list()
    for i in range(len(factors.columns)):
        name = factors.columns[i]
        X = sm.add_constant(factors[name])
        y = response
        model = sm.OLS(y, X).fit()
        vif = variance_inflation_factor(factors, i)
        info = {'name': name,
                'coef': model.params[1],
                'std_err':model.bse[1],
                't': model.tvalues[1],
                'p-value': model.pvalues[1],
                'CI_0.025': model.conf_int()[0][1],
                'CI_0.975': model.conf_int()[1][1],
                'VIF': vif}
        stats.append(info)
    df_stat = pd.DataFrame(stats)
    df_stat = df_stat.set_index('name')
        
    return df_stat


def get_selected_factor(df_factor_all, p_thres = 0.1, vif_thres = 5.0):
    """
    Input:
        df_raw: pd.DataFrame. All candidates factors and response data.
        p_thres: float. P-value threshold.
        vif_thres: float. VIF threshold.
    Output:
        df_factor: pd.DataFrame. Selected factors and response data.
        df_stat: pd.DataFrame. Statistics table for selected factors.
    """
    
    factors = df_factor_all.dropna().drop(columns = ['Log_Return_Next_1M'])
    response = df_factor_all.dropna()['Log_Return_Next_1M']
    df_stat = get_factor_stats(factors, response)
    
    #Step 1: Filter out predictors based on p-values
    df_stat = df_stat[df_stat['p-value']<=p_thres]
    df_stat['VIF'] = [variance_inflation_factor(factors[df_stat.index], i) for i in range(len(df_stat.index))]

    #Step 2: Filter out predictors based on VIF values
    vif_all = df_stat['VIF']
    while vif_all.max() > vif_thres:
        vif_all = vif_all.drop([vif_all.idxmax()])
        df = factors[vif_all.index]
        vif_all = pd.Series(data = [variance_inflation_factor(df, i) for i in range(len(vif_all.index))],
                            index = vif_all.index)

    #Update and return factor and factor stat dataframes
    df_stat = get_factor_stats(factors[vif_all.index], response)
    df_factor = df_factor_all[['Log_Return_Next_1M', 'Log_Return_1M'] + list(vif_all.index)]
    
    return df_factor, df_stat



############################################################################################################################################
#Section 3: Prediction Model    
    
def pred_ridge1(X_train, y_train, X_test, y_test, model_cv, window_len = False, message = False):
    """
    Input:
        X_train: pd.DataFrame. (n_1, p) size table where n_1 is the training sample size and p is the number of predictors.
        y_train: pd.Series. (n_1,) size series where n_1 is the training sample size.
        X_test: pd.DataFrame. (n_2, p) size table where n_2 is the test sample size and p is the number of predictors.
        y_test: pd.Series. (n_2,) size series where n_2 is the test sample size.
        model_cv: cross validation model. Example: RidgeCV.
        window_len: int or False.
            The rolling window lengh for train data. If window_len is integer, only the latest window_len data will be used.
        message: indicator for printing process update.
    Output:
        y_pred: list. Prediction results.
        alphas_fit: float. One-time fitted alpha.
        betas_fit: list. Fitted betas with intercept in the first.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_cv = scaler.transform(X_train)

    #Derive the best hyperparameter from the training set
    model_fit = model_cv.fit(X_train_cv, y_train)
    print("Best hyperparameters searching process completed for Ridge Regression.")
    
    alpha_fit = model_fit.alpha_
    if alpha_fit == max(model_cv.alphas) or alpha_fit == min(model_cv.alphas):
        print(f"Fitted alpha is on the boundary({alpha_fit}). Please check.")
    model = Ridge(alpha = alpha_fit)

    y_pred = []
    betas_fit = []
    for i in range(X_test.shape[0]):
        start = time.time()
        # update the training set with new row
        X_train_new = X_train.append(X_test.iloc[:i,:])
        y_train_new = y_train.append(y_test[:i]).to_numpy()

        # rolling window prediction
        if int(window_len) != 0:
            X_train_new = X_train_new[-window_len:,:]
            y_train_new = y_train_new[-window_len:,]

        # standardize training set and test set based only on training test
#         scaler = StandardScaler().fit(X_train_new)
        X_train_new = scaler.transform(X_train_new)
        X_test_new = scaler.transform(X_test.iloc[i:i+1, :])

        # fit and predict
        model.fit(X_train_new, y_train_new)
        pred = model.predict(X_test_new)

        # output the result
        y_pred.append(pred[0])
        betas_fit.append(np.insert(model.coef_, 0, model.intercept_))

        stop = time.time()
        duration = stop - start
        if i%10 == 0 and message:
            print(f"Loop {i+1}/{X_test.shape[0]}. Loop time: {duration}.")
            
    betas_fit = np.array(betas_fit)
    scaler_info = {'mean':scaler.mean_, 'scale': scaler.scale_}
    
    return y_pred, alpha_fit, betas_fit, scaler_info


def pred_tree1(X_train, y_train, X_test, y_test, model_cv, window_len = False, message = False):
    """
    Input:
        X_train: pd.DataFrame. (n_1, p) size table where n_1 is the training sample size and p is the number of predictors.
        y_train: pd.Series. (n_1,) size series where n_1 is the training sample size.
        X_test: pd.DataFrame. (n_2, p) size table where n_2 is the test sample size and p is the number of predictors.
        y_test: pd.Series. (n_2,) size series where n_2 is the test sample size.
        model_cv: cross validation model. Example: GridSearchCV()
        window_len: int or False.
            The rolling window lengh for train data. If window_len is integer, only the latest window_len data will be used.
        message: indicator for printing process update.
    Output:
        y_pred: list. Prediction results.
        alphas_fit: float. One-time fitted alpha.
        betas_fit: list. Fitted betas with intercept in the first.
    """

    #Derive the best hyperparameter from the training set
    model_fit = model_cv.fit(X_train, y_train)
    print("Best hyperparameters searching process completed for Random Forest.")
    best_params = model_fit.best_params_
    model = RandomForestRegressor(**best_params)

    y_pred = []
    for i in range(X_test.shape[0]):
        start = time.time()
        # update the training set with new row
        X_train_new = X_train.append(X_test.iloc[:i,:])
        y_train_new = y_train.append(y_test[:i]).to_numpy()

        # rolling window prediction
        if int(window_len) != 0:
            X_train_new = X_train_new[-window_len:,:]
            y_train_new = y_train_new[-window_len:,]

        X_test_new = X_test.iloc[i:i+1, :]

        # fit and predict
        model.fit(X_train_new, y_train_new)
        pred = model.predict(X_test_new)

        # output the result
        y_pred.append(pred[0])

        stop = time.time()
        duration = stop - start
        if i%10 == 0 and message:
            print(f"Loop {i+1}/{X_test.shape[0]}. Loop time: {duration}.")
    
    return y_pred, best_params


def pred_svr1(X_train, y_train, X_test, y_test, model_cv, window_len = False, message = False):
    """
    Input:
        X_train: pd.DataFrame. (n_1, p) size table where n_1 is the training sample size and p is the number of predictors.
        y_train: pd.Series. (n_1,) size series where n_1 is the training sample size.
        X_test: pd.DataFrame. (n_2, p) size table where n_2 is the test sample size and p is the number of predictors.
        y_test: pd.Series. (n_2,) size series where n_2 is the test sample size.
        model: linear prediction model from sklearn.
        window_len: int or False.
            The rolling window lengh for train data. If window_len is integer, only the latest window_len data will be used.
        message: indicator for printing process update.
        return_alpha: whether or not store the alpha for regularized regression.
    Output:
        y_pred: list. Prediction results.
        alphas_fit: list. Fitted alphas.
        betas_fit: list. Fitted betas with intercept in the first.
    """
    scaler = StandardScaler().fit(X_train)
    X_train_cv = scaler.transform(X_train)
    
    #Derive the best hyperparameter from the training set
    model_fit = model_cv.fit(X_train_cv, y_train)
    best_params = model_fit.best_params_
    print("Best hyperparameters searching process completed for SVR.")
    
    fitted_c = best_params['C']
    fitted_epsilon = best_params['epsilon']
    range_c = model_cv.get_params()['param_grid']['C']
    range_epsilon = model_cv.get_params()['param_grid']['epsilon']
    if fitted_c == max(range_c) or fitted_c == min(range_c):
        print(f"Fitted C is on the boundary({fitted_c}). Please check.")
    if fitted_epsilon == max(range_epsilon) or fitted_epsilon == min(range_epsilon):
        print(f"Fitted C is on the boundary({fitted_c}). Please check.")
    
    model = SVR(**best_params)
    
    y_pred = []
    for i in range(X_test.shape[0]):
        start = time.time()
        # update the training set with new row
        X_train_new = X_train.append(X_test.iloc[:i,:])
        y_train_new = y_train.append(y_test[:i]).to_numpy()

        # rolling window prediction
        if int(window_len) != 0:
            X_train_new = X_train_new[-window_len:,:]
            y_train_new = y_train_new[-window_len:,]

        # standardize training set and test set based only on training test
#         scaler = StandardScaler().fit(X_train_new)
        X_train_new = scaler.transform(X_train_new)
        X_test_new = scaler.transform(X_test.iloc[i:i+1, :])
        
        # fit and predict
        model.fit(X_train_new, y_train_new)
        pred = model.predict(X_test_new)

        # output the result
        y_pred.append(pred[0])
        
        stop = time.time()
        duration = stop - start
        if i%10 == 0 and message:
            print(f"Loop {i+1}/{X_test.shape[0]}. Loop time: {duration}.")
    scaler_info = {'mean':scaler.mean_, 'scale': scaler.scale_}
        
    return y_pred, best_params, scaler_info



def get_pred_result(X_train, y_train, X_test, y_test):
    
    tscv = TimeSeriesSplit(n_splits=5)
    alphas_ridge = np.logspace(1, 3, 61)
    param_search_svr = {'kernel': ['linear'],
                        'C': np.logspace(-5, -3, 21),
                        'epsilon': np.logspace(-3, -1, 21)}
    param_search_tree = {'criterion': ['squared_error'],
                         'max_depth': [2,3,4],
                         'min_samples_leaf': list(range(35, 50, 5)), 
                         'n_estimators': [int(i) for i in list(np.linspace(400, 600, 5))],
                         'random_state': [2024]}
    model_cv_ridge = RidgeCV(alphas=alphas_ridge, cv=tscv, fit_intercept=True, scoring='neg_mean_squared_error')
    model_cv_svr = GridSearchCV(estimator=SVR(),
                                cv=tscv, param_grid=param_search_svr, scoring='neg_mean_squared_error')
    model_cv_tree = GridSearchCV(estimator=RandomForestRegressor(),
                                 cv=tscv, param_grid=param_search_tree, scoring = 'neg_mean_squared_error')

    try:
        start = time.time()
        y_pred_ridge, alpha_fit_ridge, betas_fit_ridge, scaler_ridge = pred_ridge1(X_train, y_train, X_test, y_test, model_cv_ridge)
        stop = time.time()
        duration = stop - start
        print(f"Ridge regression prediction completed. Run time:{duration}.")
        
        start = time.time()
        y_pred_svr, best_params_svr, scaler_svr = pred_svr1(X_train, y_train, X_test, y_test, model_cv_svr)
        stop = time.time()
        duration = stop - start
        print(f"SVR prediction completed. Run time:{duration}.")

        start = time.time()
        y_pred_tree, best_params_tree = pred_tree1(X_train, y_train, X_test, y_test, model_cv_tree)
        stop = time.time()
        duration = stop - start
        print(f"Random Forest prediction completed. Run time:{duration}.")
        
    except:
        print("Error when predicting for test set. Please check.")
    
    prediction = np.array([y_pred_ridge, y_pred_svr, y_pred_tree]).T
    model_names =  ['Ridge Regression', 'SVR', 'Random Forest']
    df_pred = pd.DataFrame(data = prediction, index = X_test.index, columns = model_names)
    
    best_params = dict()
    best_params['ridge'] = {'alpha': alpha_fit_ridge, 'fitted_betas': betas_fit_ridge, 'scaler': scaler_ridge}
    best_params_svr['scaler'] = scaler_svr
    best_params['svr'] = best_params_svr
    best_params['tree'] = best_params_tree

    return df_pred, best_params



############################################################################################################################################
#Section 4: Strategy Backtest

def get_pred_performance(df_pred, y_test):
    df_pred2 = df_pred[:-1]

    pred_mse = np.mean(df_pred2.sub(y_test,axis=0)**2)
    pred_accu = np.mean(df_pred2.mul(y_test,axis=0)>0)

    matrix_ridge = confusion_matrix(y_test<0, df_pred2['Ridge Regression']<0)
    matrix_tree = confusion_matrix(y_test<0, df_pred2['Random Forest']<0)
    matrix_svr = confusion_matrix(y_test<0, df_pred2['SVR']<0)
    pred_accu_pos = [(matrix_ridge/matrix_ridge.sum(axis = 0)).diagonal()[0],
                     (matrix_tree/matrix_tree.sum(axis = 0)).diagonal()[0],
                     (matrix_svr/matrix_svr.sum(axis = 0)).diagonal()[0]]
    pred_accu_neg = [(matrix_ridge/matrix_ridge.sum(axis = 0)).diagonal()[1],
                     (matrix_tree/matrix_tree.sum(axis = 0)).diagonal()[1],
                     (matrix_svr/matrix_svr.sum(axis = 0)).diagonal()[1]]

    pred_corr = df_pred2.iloc[:-1,:].corrwith(y_test)
    pred_next = df_pred.iloc[-1, :]

    d = {'prediction_mse': pred_mse,
         'prediction_r_squared': pred_corr**2,
         'prediction_direction_accuracy': pred_accu,
         'positive_prediction_accuracy': pred_accu_pos,
         'negative_prediction_accuracy': pred_accu_neg,
         'prediction_202402': pred_next}
    
    df_pred_performance = pd.DataFrame(data=d).T
    
    return df_pred_performance


def get_strat_performance(df_factor, df_pred, y_test):
    y_ref = df_factor['Log_Return_1M']
    y_ref.name = 'S&P 500 Index'

    df_return = np.sign(df_pred.iloc[:-1, :]).mul(y_test, axis = 0).shift(1, freq = 'M')
    df_return = df_return.merge(y_ref, left_index = True, right_index = True)
    df_return = df_return[[y_ref.name] + list(df_return.columns[:-1])]
    df_return_cum = (np.exp(df_return.cumsum()))

    #exclude last month
    annualized_return = np.exp(df_return.mean()*12)-1
    annualized_excess_return = annualized_return - annualized_return[0]
    annualized_vol = np.exp(df_return.std()*np.sqrt(12))-1

    df_total = df_return_cum
    df_cummax = df_total.cummax()
    max_drawdown = -(df_total/df_cummax-1).min()

    sharpe_ratio = annualized_return/annualized_vol
    calmar_ratio = annualized_return/max_drawdown

    value_at_risk = -np.percentile(np.exp(df_return)-1, 5, interpolation="lower", axis = 0)

    d = {'annualized_return': annualized_return,
         'annualized_excess_return': annualized_excess_return,
         'annualized_volatility': annualized_vol,
         'sharpe_ratio': sharpe_ratio,
         'maximum_drawdown': max_drawdown,
         'calmar_ratio': calmar_ratio,
         'monthly_95pct_VaR': value_at_risk}
    df_strat_performance = pd.DataFrame(data=d).T
    
    return df_strat_performance, df_return, df_return_cum


