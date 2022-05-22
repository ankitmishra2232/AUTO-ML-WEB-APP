import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

def dataset_overview(df):  #data information
    shape = df.shape
    rows, cols = shape[0], shape[1]
    missing_values_sum = df.isnull().sum().sum()
    desc_stat = df.describe().to_string().split('\n')
    return rows, cols, missing_values_sum, desc_stat

def linear_regression(X_train, X_test, y_train, y_test):
    lin_reg = LinearRegression()
    desc = lin_reg.fit(X_train, y_train)
    pred = lin_reg.predict(X_test)
    lin_cdf = pd.DataFrame({'Features': X_train.columns,
                        'Coefficients': lin_reg.coef_
                        }).sort_values(by='Coefficients', ascending=False).head(10)
    return desc, pred, lin_cdf

def decision_tree_regression(X_train, X_test, y_train, y_test):
    dt_reg = DecisionTreeRegressor()
    desc = dt_reg.fit(X_train, y_train)
    pred = dt_reg.predict(X_test)
    dt_imp =  pd.DataFrame({'Features': X_train.columns,
                            'Importance': dt_reg.feature_importances_
                            }).sort_values(by='Importance', ascending=False).head(10)
    return desc, pred, dt_imp

def random_forest_regression(X_train, X_test, y_train, y_test):
    rf_reg = RandomForestRegressor()
    desc = rf_reg.fit(X_train, y_train)
    pred = rf_reg.predict(X_test)
    rf_imp =  pd.DataFrame({'Features': X_train.columns,
                            'Importance': rf_reg.feature_importances_
                            }).sort_values(by='Importance', ascending=False).head(10)
    return desc, pred, rf_imp

def logistic_regression(X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression()
    desc = log_reg.fit(X_train, y_train)
    pred = log_reg.predict(X_test)
    log_cdf =  pd.DataFrame({'Features': X_train.columns,
                            'Coefficients': log_reg.coef_[0]
                            }).sort_values(by='Coefficients', ascending=False).head(10)
    return desc, pred, log_cdf

def decision_tree_classifier(X_train, X_test, y_train, y_test):
    dt_clf = DecisionTreeClassifier()
    desc = dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    dtc_imp =  pd.DataFrame({'Features': X_train.columns,
                            'Importance': dt_clf.feature_importances_
                            }).sort_values(by='Importance', ascending=False).head(10)
    return desc, pred, dtc_imp

def random_forest_classifier(X_train, X_test, y_train, y_test):
    rf_clf = RandomForestClassifier()
    desc = rf_clf.fit(X_train, y_train)
    pred = rf_clf.predict(X_test)
    rfc_imp =  pd.DataFrame({'Features': X_train.columns,
                            'Importance': rf_clf.feature_importances_
                            }).sort_values(by='Importance', ascending=False).head(10)
    return desc, pred, rfc_imp

def regression_metrics(y_test, pred):
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    return mae, mse, rmse, r2

def classification_metrics(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(y_test, pred)
    class_rep = classification_report(y_test, pred, output_dict=True)
    return accuracy, conf_mat, class_rep