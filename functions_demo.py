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
def source_yfinance(start_date, end_date):
    """
    Input:
        start_date: string type. Example: "1990-01-01".
        end_date: string type. Example: "2024-01-24".
    Output:
        df_yfdata: pd.DataFrame type.
            The first column Log_Return_Next_1M is the independent varialble.
    """
    try:
        # Get monthly return data and calculate intra-month return.
        df_yfdata = yf.download(tickers = '^GSPC', interval = "1mo", start = start_date, end = end_date)
        df_yfdata = df_yfdata.sort_index()
        df_yfdata['return_next_intramonth'] = np.log(df_yfdata['Close']/df_yfdata['Open']).shift(-1)
        df_yfdata['return_current_month'] = np.log(df_yfdata['Close']/df_yfdata['Close'].shift(1))

        # Get daily return data and calculate exponential weighted moving average prices.
        df_yfdata3 = yf.download(tickers = '^VIX', interval = "1d", start = start_date, end = end_date).resample('MS').last()
        df_yfdata['VIX_Close'] = df_yfdata3['Close']

        # Select needed columns and rows and return the dataframe
        df_yfdata = df_yfdata[['return_next_intramonth', 'return_current_month', 'Close', 'Volume', 'VIX_Close']]
        df_yfdata.index = df_yfdata.index.to_period('M').to_timestamp('M')

        print(f"Start date {df_yfdata.index[0]}. End date {df_yfdata.index[-1]}. Total data points {df_yfdata.shape[0]}.")
        print("yfinance data sourced successfully.")
        return df_yfdata
    
    except:
        print("Error when sourcing data from yfinance. Please check source_yfinance.")


def get_mp_ts(url, pattern = '†\n'):
    """
    Input:
        url: str.
        pattern: str.
    Output:
        ts: pandas series with date as index.
    """
    data = requests.get(url).text
    soup = BeautifulSoup(data, 'html.parser')
    table = soup.find("table", id = "datatable")
    date = []
    values = []
    
    if url == "https://www.multpl.com/s-p-500-historical-prices/table/by-month":
        for row in table.find_all("tr")[2:]:
            date.append(datetime.strptime(row.find_all("td")[0].text.strip(), "%b %d, %Y"))
            temp_lst = row.find_all("td")[1].text.strip('\n\u2002\n').split(',')
            if len(temp_lst) == 2:
                num = float(temp_lst[0])*1000 + float(temp_lst[1])
            elif len(temp_lst) == 1:
                num = float(temp_lst[0])
            values.append(num)
            ts = pd.Series(data = values, index = date)
    else:
        for row in table.find_all("tr")[2:]:
            date.append(datetime.strptime(row.find_all("td")[0].text.strip(), "%b %d, %Y"))
            values.append(float(row.find_all("td")[1].text.strip(pattern)))
        ts = pd.Series(data = values, index = date)
    
    return ts


def source_multpl(start_date, end_date):
    """
    Input:
        start_date: string type. Example: "1990-01-01".
        end_date: string type. Example: "2024-01-24".
    Output:
        df_mpdata: pd.DataFrame type.
    Notes:
        This function will use get_mp_ts to source data from multpl.com.
    """
    try:
        ts1 = get_mp_ts("https://www.multpl.com/s-p-500-historical-prices/table/by-month")
        ts2 = get_mp_ts("https://www.multpl.com/s-p-500-pe-ratio/table/by-month")
        ts4 = get_mp_ts("https://www.multpl.com/s-p-500-dividend-yield/table/by-month", '†\n%')

        # Data Cleaning
        ts1 = ts1.sort_index().dropna()
        ts2 = ts2.sort_index().dropna()
        ts4 = ts4.sort_index().dropna()
        ts1.index = ts1.index.to_period('M').to_timestamp('M')
        ts2.index = ts2.index.to_period('M').to_timestamp('M')
        ts4.index = ts4.index.to_period('M').to_timestamp('M')

        # Convert pd series to pd Dataframe
        df_mpdata = pd.DataFrame({'PE': ts2, 'Earnings': ts1/ts2, 'Div_Yield': ts4})
        df_mpdata = df_mpdata.loc[df_mpdata.index >= start_date]
        df_mpdata = df_mpdata.loc[df_mpdata.index <= end_date]
        

        print(f"Start date {df_mpdata.index[0]}. End date {df_mpdata.index[-1]}. Total data points {df_mpdata.shape[0]}.")
        print("Multpl data sourced successfully.")
        return df_mpdata
    
    except:
        print("Error when sourcing data from multpl. Please check source_multpl.")


def source_fred(start_date, end_date):
    """
    Input:
        start_date: string type. Example: "1990-01-01".
        end_date: string type. Example: "2024-01-24".
    Output:
        df_fddata: pd.DataFrame type.
    Notes:
        This function will use web.DataReader to source data from FRED.
    """
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') - relativedelta(months=1)
        #monthly data
        inf_factors = ['CPIAUCSL', #CPI All items
                       'CPIUFDSL', #CPI Food
                           'CUSR0000SAF11', 'CUSR0000SEFV',
                       'CPIENGSL', #CPI Energy
                           'CUSR0000SACE', #CPI Energy Commodities
                               'CUSR0000SETB01', 'CUSR0000SEHE',
                           'CUSR0000SEHF', #CPI Energy Services
                               'CUSR0000SEHF01', 'CUSR0000SEHF02',
                       'CPILFESL', #CPI All Items Less Food and Energy
                           'CUSR0000SACL1E', #CPI Commodities Less Food and Energy Commodities
                                'CUUR0000SETA01', 'CUSR0000SETA02', 'CPIAPPNS', 'CUSR0000SAM1',
                           'CUSR0000SASLE', #CPI Services Less Energy Services
                                'CUSR0000SAH1', 'CUUR0000SAS4', 'CUSR0000SAM2',
                       'PCUOMFGOMFG', 'PCUOMINOMIN', 'DCOILWTICO'] #DCOILWTICO is daily
        df_monthly = web.DataReader(inf_factors[:-1] + ['PSAVERT', 'FEDFUNDS'],'fred', start = start_dt, end = end_date)
        df_monthly.index = df_monthly.index.to_period('M').to_timestamp('M').shift(1)
        df_monthly['FEDFUNDS'] = df_monthly['FEDFUNDS'].shift(-1)        # shift Fed Funds Rate backward since it is updated daily

        #daily data
        df_daily = web.DataReader(['DGS1'] + [inf_factors[-1]], 'fred', start = start_dt, end = end_date).resample('M').last()

        #merge together
        df_fddata = df_monthly.merge(df_daily, how = 'outer', left_index = True, right_index = True)    
        df_fddata = df_fddata.iloc[1:]
        
        df_fddata = df_fddata.loc[df_fddata.index >= start_date]
        df_fddata = df_fddata.loc[df_fddata.index <= end_date]

        print(f"Start date {df_fddata.index[0]}. End date {df_fddata.index[-1]}. Total data points {df_fddata.shape[0]}.")
        print("FRED data sourced successfully.")
        return df_fddata, inf_factors
    
    except:
        print("Error when sourcing data from FRED. Please check source_fred.")


def source_sentiment(start_date, end_date, filename = "tbmpx1px5.csv"):
    """
    Input:
        start_date: string type. Example: "1990-01-01".
        end_date: string type. Example: "2024-01-24".
        file_name: string type. Example: "tbmpx1px5.csv"
    Output:
        df_stdata: pd.DataFrame type.
    Notes:
        This function will read data from downloaded Expected Changes in Inflation Rates file from http://www.sca.isr.umich.edu/tables.html.
    """
    try:
        df_stdata = pd.read_csv(filename)
        df_stdata.index = pd.to_datetime(df_stdata['Month'] + df_stdata['YYYY'].astype(str)) + pd.offsets.MonthEnd()
        df_stdata = df_stdata.drop(columns = ['Month', 'YYYY'])
        df_stdata = df_stdata.loc[df_stdata.index >= start_date]
        df_stdata = df_stdata.loc[df_stdata.index <= end_date]
        
        print(f"Start date {df_stdata.index[0]}. End date {df_stdata.index[-1]}. Total data points {df_stdata.shape[0]}.")
        print("Sentiment data sourced successfully.")
        return df_stdata
    
    except:
        print(f"Error when sourcing data from {filename}. Please check source_sentiment.")


def source_raw_data(start_date, end_date, filename = "tbmpx1px5.csv"):
    """
    Input:
        start_date: string type. Example: "1990-01-01".
        end_date: string type. Example: "2024-01-24".
        file_name: string type. Example: "tbmpx1px5.csv"
    Output:
        df_data: pd.DataFrame type.
    Notes:
        This function will read data from downloaded Expected Changes in Inflation Rates file from http://www.sca.isr.umich.edu/tables.html.
    """
    try:
        #source data
        data_yf = source_yfinance(start_date, end_date)
        data_mp = source_multpl(start_date, end_date)
        data_fd, inf_factors = source_fred(start_date, end_date)
        data_st = source_sentiment(start_date, end_date, filename = "tbmpx1px5.csv")

        #merge data
        df_raw = data_yf.merge(data_mp, how = 'outer', left_index = True, right_index = True)
        df_raw = df_raw.merge(data_fd, how = 'outer', left_index = True, right_index = True) 
        df_raw = df_raw.merge(data_st, how = 'outer', left_index = True, right_index = True)

        return df_raw, inf_factors
    
    except:
        print("Error when merging data. Please check source_raw_data.")


def generate_factor_data(df_raw, inf_factors):
    """
    Input:
        df_raw: pd.DataFrame. Raw predictors and response data.
    Output:
        df_factor_all: pd.DataFrame. Transformed factors and response data.
    """

    #Response/dependent variables
    df_factor_all = pd.DataFrame(index = df_raw.index)
    df_factor_all.index.names = ['Date']
    df_factor_all['return_next_intramonth'] = df_raw['return_next_intramonth']
    df_factor_all['return_current_month'] = df_raw['return_current_month']

    #Predictors/independent variables
    #Economic factors
    inf_names = ['fctr_inflation' + str(i+1) for i in range(len(inf_factors))]
    df_factor_all[inf_names] = np.log(df_raw[inf_factors]/df_raw[inf_factors].shift(12))
    df_factor_all['fctr_personalsaving'] = np.log(df_raw['PSAVERT'])
    df_factor_all['fctr_fedfunds_spread'] = (df_raw['FEDFUNDS'] - df_raw['DGS1']).rolling(window = 12).sum()

    #Sentiment factors
    df_factor_all['fctr_inflation_exp'] = np.log(df_raw['PX_MD']/df_raw['PX_MD'].shift(12))

    #Valuation factors
    df_factor_all['fctr_pe_ratio'] = 1/df_raw['PE']
    df_factor_all['fctr_gep_ratio'] = np.log(df_raw['Earnings']/df_raw['Earnings'].shift(1))/df_raw['PE']
    df_factor_all['fctr_div_yield'] = np.log(df_raw['Div_Yield'])

    #Technical factors
    Volume_TTM = df_raw['Volume'].rolling(window = 12).sum()
    df_factor_all['fctr_volumn_return'] = np.log(Volume_TTM/Volume_TTM.shift(1))*df_raw['return_current_month']
    df_factor_all['fctr_vix'] = np.log(df_raw['VIX_Close']/df_raw['VIX_Close'].shift(6))

    df_factor_all = pd.concat([df_factor_all.dropna(), df_factor_all.tail(1)])
    
    
    return df_factor_all



############################################################################################################################################
#Section 2: Feature Selection
def select_factor1(df_factors, response, num_features = 5, alphas = np.linspace(0.0001, 0.01, 991)):#np.linspace(0.001, 0.01, 91)):
    """
    Input:
        df_raw: pd.DataFrame. This should contain columns with inf_factors and resp_name
        inf_factors: list. A list of strings that contains all candidate factors' names.
        resp_name: string. Name of the response variable.
        num_features: int. Maximum number of 
    Output:
        feature_sel: pandas.core.indexes.base.Index. Contains selected feature names from Lasso Regression.
    Notes
    """
    X = (df_factors - df_factors.mean(axis = 0))/df_factors.std(axis = 0)
    X = sm.add_constant(X)
    y = response

    #Use Lasso to select the top 5 inflation factors
    for alpha in alphas:
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        if np.count_nonzero(lasso.coef_[1:]) == num_features:
            break
        elif np.count_nonzero(lasso.coef_[1:]) < num_features:
            print(f"Lasso selected fewer features than desired. Please check select_factor.")
            break
        else:
            continue

    feature_ind = np.nonzero(lasso.coef_[1:])[0]
    factors_sel = X.columns[1:][feature_ind]

    return factors_sel


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
        info = {'Factor Name': name,
                'beta': model.params[1],
                'std. error':model.bse[1],
                't-score': model.tvalues[1],
                'p-value': model.pvalues[1],
                'CI_0.025': model.conf_int()[0][1],
                'CI_0.975': model.conf_int()[1][1],
                'VIF': vif}
        stats.append(info)
    df_stat = pd.DataFrame(stats)
    df_stat = df_stat.set_index('Factor Name')
        
    return df_stat


def select_factor2(df_factors, response, p_thres = 0.05, vif_thres = 2.0):
    """
    Input:
        df_raw: pd.DataFrame. All candidates factors and response data.
        p_thres: float. P-value threshold.
        vif_thres: float. VIF threshold.
    Output:
        factors_sel: list. A list of selected factors based on p values and VIF.
    """
    df_stat = get_factor_stats(df_factors, response)

    #Step 1: Filter out predictors based on p-values
    df_stat = df_stat[df_stat['p-value']<=p_thres]
    df_stat['VIF'] = [variance_inflation_factor(df_factors[df_stat.index], i) for i in range(len(df_stat.index))]

    #Step 2: Filter out predictors based on VIF values
    vif_all = df_stat['VIF']
    while vif_all.max() > vif_thres:
        vif_all = vif_all.drop([vif_all.idxmax()])
        df = df_factors[vif_all.index]
        vif_all = pd.Series(data = [variance_inflation_factor(df, i) for i in range(len(vif_all.index))],
                            index = vif_all.index)

    #Update and return factor and factor stat dataframes
    factors_sel = list(vif_all.index)
    
    return factors_sel



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
            X_train_new = X_train_new.iloc[-window_len:,:]
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
            X_train_new = X_train_new.iloc[-window_len:,:]
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
            X_train_new = X_train_new.iloc[-window_len:,:]
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



def get_pred_result(X_train, y_train, X_test, y_test, alphas_ridge, param_search_svr, param_search_tree, window_len = False):
    
    tscv = TimeSeriesSplit(n_splits=5)
    model_cv_ridge = RidgeCV(alphas=alphas_ridge, cv=tscv, fit_intercept=True, scoring='neg_mean_squared_error')
    model_cv_svr = GridSearchCV(estimator=SVR(),
                                cv=tscv, param_grid=param_search_svr, scoring='neg_mean_squared_error')
    model_cv_tree = GridSearchCV(estimator=RandomForestRegressor(),
                                 cv=tscv, param_grid=param_search_tree, scoring = 'neg_mean_squared_error')

    try:
        start = time.time()
        y_pred_ridge, alpha_fit_ridge, betas_fit_ridge, scaler_ridge = pred_ridge1(X_train, y_train, X_test, y_test, model_cv_ridge, window_len)
        stop = time.time()
        duration = stop - start
        print(f"Ridge regression prediction completed. Run time:{duration}.")
        
        start = time.time()
        y_pred_svr, best_params_svr, scaler_svr = pred_svr1(X_train, y_train, X_test, y_test, model_cv_svr, window_len)
        stop = time.time()
        duration = stop - start
        print(f"SVR prediction completed. Run time:{duration}.")

        start = time.time()
        y_pred_tree, best_params_tree = pred_tree1(X_train, y_train, X_test, y_test, model_cv_tree, window_len)
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

    pred_accu = np.mean(df_pred2.mul(y_test,axis=0)>0)
    conf_matrices = np.array([confusion_matrix(y_test>0, df_pred2[model_name]>0) for model_name in df_pred2.columns])
    pred_precision = np.array([cm[1,1]/cm[:,1].sum() for cm in conf_matrices])
    pred_recall = np.array([cm[1,1]/cm[1,:].sum() for cm in conf_matrices])
    pred_f1 = 2*pred_precision*pred_recall/(pred_precision+pred_recall)
    
    pred_corr = df_pred2.iloc[:,:].corrwith(y_test)
    pred_mse = np.mean(df_pred2.sub(y_test,axis=0)**2)
    
    pred_next = np.array(['Positive' if i>=0 else 'Negative' for i in df_pred.iloc[-1, :]])
    next_tmsp = df_pred.index[-1] + pd.DateOffset(months=1)
    next_mth = pd.to_datetime(next_tmsp).strftime('%b-%Y') 

    d = {'Accuracy': pred_accu,
         'Precision': pred_precision,
         'Recall': pred_recall,
         'F1 Score': pred_f1,
#          'MSE': pred_mse,
#          'R-squared': (pred_corr**2)*np.sign(pred_corr),
         'Next Month': np.repeat(next_mth, 3),
#          'Predicted Return': df_pred.iloc[-1, :],
         'Predicted Return Direction': pred_next}

    df_pred_performance = pd.DataFrame(data=d).T
    
    return df_pred_performance, conf_matrices



def get_strat_performance(df_pred, y_ref, y_test):
    df_pred2 = df_pred[:-1]
    y_ref.name = 'S&P 500 Index'

    df_return = np.sign(df_pred2).mul(y_test, axis = 0).shift(1, freq = 'M')
    df_return = df_return.merge(y_ref, left_index = True, right_index = True)
    df_return = df_return[[y_ref.name] + list(df_return.columns[:-1])]
    df_return_cum = (np.exp(df_return.cumsum()))

    #exclude last month
    annualized_return = np.exp(df_return.mean()*12)-1
    annualized_active_return = annualized_return - annualized_return[0]
    annualized_vol = np.exp(df_return.std()*np.sqrt(12))-1

    df_total = df_return_cum
    df_cummax = df_total.cummax()
    max_drawdown = -(df_total/df_cummax-1).min()

    sharpe_ratio = annualized_return/annualized_vol
    calmar_ratio = annualized_return/max_drawdown

    value_at_risk = -np.percentile(np.exp(df_return)-1, 5, interpolation="lower", axis = 0)

    win_rate = (df_return>0).mean()
    long_months = pd.concat([pd.Series([df_return.shape[0]], index = [df_return.columns[0]]), (df_pred2>0).sum()])
    short_months = pd.concat([pd.Series([0], index = [df_return.columns[0]]), (df_pred2<0).sum()])
    position_change = [0]+[((df_pred2[model_name]*df_pred2[model_name].shift(1))<0).sum() for model_name in df_pred2.columns]
    slugging = [-df_return[model_name][df_return[model_name]>0].mean()/df_return[model_name][df_return[model_name]<0].mean()  
                for model_name in df_return.columns]

    d = {'Annualized Return': annualized_return,
         'Annualized Active Return': annualized_active_return,
         'Annualized Vol': annualized_vol,
         'Max Drawdown': max_drawdown,
         '5% VaR': value_at_risk,
         'Sharpe Ratio': sharpe_ratio,
         'Calmar Ratio': calmar_ratio,
         'Win Rate': win_rate,
         'IC': win_rate*2-1,
         'Long Months': long_months,
         'Short Months': short_months,
         'Position Changes': position_change,
         'Slugging': slugging}
    df_strat_performance = pd.DataFrame(data=d).T
    
    return df_strat_performance, df_return, df_return_cum

