# S&P500 Index Long-Short Trading Strategy
In this project, we selected 8 factors for predicting S&amp;P 500 index' next-month returns with Lasso Regression and variance inflation factor (VIF). Then we applied Ridge Regression, Support Vector Regression (SVR). and Random Forest respectively for return prediction and achieved **70.37%, 61.73%, and 65.43%** accuracy in return direction prediction. Based on the prediction results, three long-short trading strategy for S&P 500 Index were built and backtested, and the performance of the strategies and S&P 500 Index in time-series are shown in Figure 1.

![alt text](plots/figure1_strategy_performance.png)

**Please notice that this project is for demonstration only, and it does not provide any investment advice.** <br />
All data and code are available at the [repository](https://github.com/michaelli99/1.S-P500-Index-Long-Short-Strategy) for replication purpose. <br />
The following content is divided into five parts to demonstrate the process and performance of the prediction models, and the project's workflow is shown by the following diagram:

```mermaid
flowchart TD
    A[Data Sourcing] --> B[Feature Selection]
    B --> C[Ridge Regression]
    B --> D[SVR]
    B --> E[Random Forest]
    C --> F[Prediction and Strategy Performance Evaluation]
    D --> F
    E --> F
    F --> G["Prediction Attribution (Ridge Regression)"]
```

## 1. Data Sourcing
In this project, we sourced all the data from online databases such as Yahoo Finance and FRED. All the indices and factors’ raw data falls into the period of July 1990 to December 2023.

### 1.1. Response/target variable:  
The target varialble of prediction is **S&P 500 Index's next intra-month log returns**. The intra-month log return in month i is calculated with the formula: $$y_i = log(\frac{P_{close, i}}{P_{open, i}})$$
We chose to use log return because of its potential of being normally distributed, and we used intra-month return so that the prediction can be transformed into actionable trading signal.
   
### 1.2. Predictors/independent variables:  
To predict the target variable, we first built a pool of candidate regressors with raw predictors data and basic mathematical transformation. The raw data can be classified into three categories: **economic, fundamental, and technical data**. Below is a short description for each category.
- Economic data includes macroeconomic indicators such as CPI components, employment statistics, and interest rates. Most of them are related to monetary or fiscal policy.
- Fundamental data consists of valuation data for S&P 500 Index such as earnings, PE, and dividend yield.
- Technical data was derived from S&P 500 Index and VIX's historical price and trading volume.

After sourcing the data, we converted all factors data into monthly basis. Then we moved historical data to the actual data release month to prevent data leakage. Finally, all response and predictors' monthly data are available from July 1990 to December 2023 with a total of 402 months.

## 2. Feature Engineering
Raw predictors' data were transformed into 35 candidate regressors using basic mathematical operations. After factor transformation, we applied a two-step factor selection process to select the most significant regressors for predicting the target variable:
- Step1: Select up to 5 regressors from each sub-category using Lasso regression.
- Step2: Select the most significant regressors from all categories based on t-score and variance inflation factor (VIF) thresholds.

After the two above steps, 8 regressors were selected from the feature engineering process. The summary statistics for the 8 selected regressors is shown below:

![alt text](plots/dataframe1_factor_stat.png)

The 8 selected regressors consist of 5 macroeconomic factors, 1 fundamental factor, and 2 technical factors. All regressors are continuous variables except for EWMA_Cross_Ind1, which is a binary variable that only takes value from 1 and 0. Shown below are the regressors' distribution and time series plots: 

![alt text](plots/figure2_factors_dist.png)

![alt text](plots/figure3_factors_ts.png)

All of the following prediction models are based on the assumption/prior of **the 8 selected factors' association with S&P 500 Index's next month return is not changed**.

## 3. Models Training and Testing
To predict the target variable, we applied three different machine learning models: **Ridge Regression, Support Vector Regression (SVR), and Random Forest**.

The data were divided into training and testing set with the classic 80-20 split. The original sequece of the data was maintained, and we adopted one-month ahead prediction in the testing set.

### 3.1. Training set
**Training set data spans from 1990-07-31 to 2017-03-31 with a total of 321 data points.** In the training set, we derived the best hyperparameters for each prediction model. Additionally, since there were regularization/penalization components in ridge regression and support vector regression models, regressors had to be standardized/normalized to achieve equal importance in the prediction. Hence the training set was also used to derive the nomralization scalar.  
Below is a summary of hyperparameters that were derived from the trainings set:
- **Ridge Regression:**
    - Alpha: Constant that multiplies the L2 term, controlling regularization strength.
    - Normalization scalar: $\mu$ and $\sigma$.
- **Support Vector Regression:**
    - C: Regularization parameter that inversely relates to the strength of the regularization.
    - epsilon: The epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
    - Normalization scalar: $\mu$ and $\sigma$.
- **Random Forest:**
    - The number of trees n.
    - The maximum depth of the tree.
    - The minimum number of samples required to split an internal node.
    
We applied a 5-split time-series cross validation to the training set to derive the best hyperparameters for each prediction model. After getting the best hyperparameters for each model, we used thees hyperparameters in the testing set to predict for the target variable.

### 3.2. Testing set
**Testing set data spans from 2017-04-30 to 2023-12-31 with a total of 81 data points.** In the testing set, we used training set's best hyperparameters and scalars, and we adopted one-step ahead prediction. In other words, we trained each model with all available historical data up to the current month when predicting for next month's return.  

## 4. Performance Evaluation
After training the model and collecting the prediction results, we evaluated three prediction models from two perspectives: **prediction accuracy and trading strategy's performance**.

### 4.1. Prediction performance analysis
In prediction analysis, we summarized each model's prediction mean squared error (MSE), R-squared, and accuracies of predicted return direction in the dataframe, and we also used scatterplots and histograms to visualize the predicted values and prediction errors.

![alt text](plots/dataframe2_pred_performance.png)

![alt text](plots/figure4_pred_return_plot.png)

![alt text](plots/figure5_pred_error_plot.png)

![alt text](plots/figure6_pred_error_hist.png)

From the above summary statistics table and plots, we have the following observations for each prediction model:
**1. SVR  **
SVR model achieved the lowest MSE (0.002267) and the highest R-squared (0.139177) among all three models. Both MSE and R-squared statistics indicate SVR has the lowest prediction error square on average. The lowest MSE achieved by SVR model can also be observed from Figure 5 where prediction errors of SVR are generally distributed closer to x-axis.  
Surprisingly, SVR has the lowest prediciton direction accuracy (61.73%) among all three models. The prediction direction accuracy is calculated by dividing the frequency of predicted return and actual return have the same sign by total count of prediction. In Figure 4, the prediction direction accuracy is the proportion of Quadrants II and IV's points in the whole plot. We can see that SVR has more points in Quardrant II than the other two prediction models. If we set negative predicted return as rejecting the null hypothesis, this indicates that SVR has higher chance of being false positive (Type I error). However, we also notice that some of the misclassifed points from SVR are very close to x-axis. This can be explained by the hyperparameter of epsilon in SVR model. The value of epsilon will define a margin of tolerance where no penalty is given to error, making the model ignore small prediction errors and assigning more extreme valuese as "support vectors". Compared with Ridge Regression and Random Forest, SVR is more robust to less extreme data points and is more accurate for predicting more extreme values. In the strategy performance evaluation part, we also notice that SVR-based strategy results in better return performance despite the lowest prediciton direction accuracy (or "win ratio").  

**2. Ridge Regression  **
Ridge Regression ranks second in MSE and R-squared, and its performance is comparable with SVR. This could be explained by the fact that we use linear kernel for SVR so that both Ridge Regression and SVR predictions are based on linear transformation of the regressors. From Figure 3, we can observe that Ridge Regression often gives more conservative and less extreme predictions than SVR. This can also explained by a relatively large regularization constant (alpha) derived from the training set and the hyperparameter epsilon in SVR which ignores smallerror for SVR training.  
Ridge Regression has the best performance in return prediction's direction. The best performance in prediction accuracy can also be observed from Figure 4 where Ridge Regression appears to have the most points in Quadrants I and III.  
Ridge Regression also has the lowest prediction bias with an average prediction error of 0.001 as shown in Figure 6.

**3. Random Forest  **
Random Forest ranked last in prediction error 
Random Forest is based on ensembling decision trees, and it results in the most conservative prediction as all of the predicted values fall in the range (-0.02, 0.02) possibly because of the averaging effect of all trees. The prediction R-sqaured of Random Forest is significantly lower than the R-squared of Ridge Regression and SVR, indicating that Random Forest's predictions are very helpful in explaining the variation of the target variable.  
Random Forest ranks second in prediction accuracy and first in positive prediction accuracy. One advantage of Random Forest is that the model has a positive prediction accuracy of 75.56%, suggesting that Random Forest is most likely to be correct when it predicts the target variable to be positive.  

**4. General observation  **
The R-sqaured for all prediction models are less than 0.15, indicating that the majority of S&P 500 Index's returns are not explained by the models and selected factors. This is not surprising because our factor universe is limited, and hence it is not supposed to include all factors that could explain S&P 500 Index's future return. Also, all the factors data were based on historical events or expectations. Contingent events may happen during the target month of prediction and impact the index's return. It turns out that all the models achieved more than 60% of prediction direction accuracy with less than 0.15 R-squared.

## 5. Prediction Attribution (Ridge Regression Only)
