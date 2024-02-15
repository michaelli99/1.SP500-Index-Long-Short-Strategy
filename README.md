# S&P500 Index Long-Short Trading Strategy
In this project, we selected 8 factors for predicting S&amp;P 500 index' next-month returns with Lasso Regression and variance inflation factor (VIF). Then we applied Ridge Regression, Support Vector Regression (SVR). and Random Forest respectively for return prediction and achieved **70.37%, 61.73%, and 65.43%** accuracy in return direction prediction. Based on the prediction results, three long-short trading strategy for S&P 500 Index were built and backtested, and the performance of the strategies and S&P 500 Index in time-series are shown in Figure 1.

![alt text](plots/figure1_strategy_performance.png)

Please notice that this project is for demonstration only, and it does not provide any investment advice. <br />
All data and code are available at [this repository](https://github.com/michaelli99/1.S-P500-Index-Long-Short-Strategy) for replication purpose. <br />
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
The target varialble of prediction is **S&P 500 Index's next intra-month log returns**. The intra-month log return is calculated with the formula: $$y_i = log(\frac{P_{close, i}}{P_{open, i}})$$
We chose to use log return because of its potential of being normally distributed, and we used intra-month return so that the prediction can be transformed into actionable trading signal.
   
### 1.2. Predictors/independent variables:  
To predict the target variable, we first build a pool of candidate regressors with raw predictors data and basic mathematical transformation. The raw data can be classified into three categories: **economic, fundamental, and technical data**. Below is a short description for each category.
- Economic data includes macroeconomic indicators such as CPI components, employment statistics, and interest rates. Most of them are related to monetary/fiscal policy.
- Fundamental data consists of valuation data for S&P 500 Index such as earnings, PE, and dividend yield.
- Technical data was derived from S&P 500 Index and VIX's historical price and trading volume.
After sourcing the data, we converted all factors data into monthly basis. Then we moved historical data to the actual data release month to prevent data leakage. Finally, all response and predictors' monthly data are available from July 1990 to December 2023 with a total of 402 months.

## 2. Feature Engineering
Raw predictors data were transformed into 35 candidate regressors using intuition and basic mathematical operations. After factor transformation, we applied a two-step factor selection process to select the most significant regressors for predicting the target variable:
- Step1: Select up to 5 regressors from each sub-category using Lasso regression.
- Step2: Select the most significant regressors from all categories based on t-score and variance inflation factor (VIF) thresholds.
After the two steps, 8 regressors were selected from the feature engineering process. The 8 selected regressors consist of 5 macroeconomic factors, 1 fundamental factor, and 2 technical factors. All factors are continuous variables except for EWMA_Cross_Ind1, which is a binary variable that only takes value from 1 and 0.
The summary statistics for the 8 selected regressors is shown below:
## 3. Models Training and Testing
## 4. Prediction and Strategy Performance Evaluation
## 5. Prediction Attribution (Ridge Regression Only)
