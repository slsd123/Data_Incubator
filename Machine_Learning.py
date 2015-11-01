import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn.ensemble import RandomForestClassifier

def Machine_Learning(input, output, date_range):

  Predicted_Output = pd.DataFrame(index=date_range, columns=[])
  for Year_In_Question in date_range:
    X = np.array(input[input.index != Year_In_Question])
    x = np.array(input[input.index == Year_In_Question])
    Y = np.ravel(output[output.index != Year_In_Question])

# Theil Sen Regressor regression
    predict_ = x
    gp = linear_model.TheilSenRegressor(tol=0.0001)
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'TheilSen'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'TheilSen Percent Error'] = float(np.abs(output.loc[Year_In_Question] - y_pred)/output.loc[Year_In_Question]*100)

# Linear regression
    predict_ = x
    gp = linear_model.LinearRegression()
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'Linear'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'Linear Percent Error'] = float(np.abs(output.loc[Year_In_Question] - y_pred)/output.loc[Year_In_Question]*100)

# Polynomial
    poly     = PolynomialFeatures(degree=2)
    X_       = poly.fit_transform(X)
    predict_ = poly.fit_transform(x)
    gp = linear_model.LinearRegression()
    gp.fit(X_, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'Polynomial'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'Polynomial Percent Error'] = float(np.abs(output.loc[Year_In_Question] - y_pred)/output.loc[Year_In_Question]*100)

# Ridge Regression
    predict_ = x
    gp = linear_model.Ridge()
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'Ridge'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'Ridge Percent Error'] = float(np.abs(output.loc[Year_In_Question] - y_pred)/output.loc[Year_In_Question]*100)

# Lasso Regression
    predict_ = x
    gp = linear_model.Lasso(max_iter = 10000)
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'Lasso'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'Lasso Percent Error'] = float(np.abs(output.loc[Year_In_Question] - y_pred)/output.loc[Year_In_Question]*100)

# Elastic Net
    predict_ = x
    gp = linear_model.ElasticNet(max_iter = 10000)
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'Elastic'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'Elastic Percent Error'] = float(np.abs(output.loc[Year_In_Question] - y_pred)/output.loc[Year_In_Question]*100)

# Bayesian Ridge Modeling
    predict_ = x
    gp = linear_model.BayesianRidge()
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'BayRidge'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'BayRidge Percent Error'] = float(np.abs(output.loc[Year_In_Question] - y_pred)/output.loc[Year_In_Question]*100)

# Gaussian Processes for Machine Learning (GPML)
    predict_ = x
    gp = gaussian_process.GaussianProcess()
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'GPML'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'GPML Percent Error'] = float(np.abs(output.loc[Year_In_Question] - y_pred)/output.loc[Year_In_Question]*100)

# Random Forest Regression
    predict = x
    gp = RandomForestClassifier(n_estimators=1000)
    gp.fit(X, Y)
    y_pred = gp.predict(predict_)
    Predicted_Output.loc[Year_In_Question,'RandFor'] = float(y_pred)
    Predicted_Output.loc[Year_In_Question,'RandFor Percent Error'] = float(np.abs(output.loc[Year_In_Question] - y_pred)/output.loc[Year_In_Question]*100)

# Add the actual crop yield to the Predicted_Output DataFrame
    Predicted_Output.loc[Year_In_Question, 'Crop Yield'] = np.array(output[output.index == Year_In_Question])

  print Predicted_Output

  return Predicted_Output
