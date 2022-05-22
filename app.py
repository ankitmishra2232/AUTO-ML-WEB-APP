# import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# from flask_pymongo import PyMongo
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from model import linear_regression,logistic_regression,decision_tree_classifier,decision_tree_regression,random_forest_classifier,random_forest_regression,regression_metrics,classification_metrics,DecisionTreeClassifier,DecisionTreeRegressor,dataset_overview

app = Flask(__name__)
UPLOAD_FOLDER = 'F:/Uploads/'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file(path):  #using the file name and path data is being read here
    try:
        global df
        df = pd.read_csv(path)
        print('Success')
        return df
    except:
        print('Could not read file.')



@app.route('/results', methods=['POST'])
def show_results():
    if request.method == 'POST':
        models = request.form.getlist('models')
        test_split = float(request.form.getlist('split')[0])
        global selected_cols
        selected_cols = []
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        if y.dtype == 'object' and y.str.isdigit().all():
            y=y.astype('int')
        X = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
        model_pkg = []
        
        for model in models:
            if model == 'Linear Regression':
                desc, pred, lin_cdf = linear_regression(X_train, X_test, y_train, y_test)
                lin_cdf = lin_cdf.to_html(classes='data table')
                mae, mse, rmse, r2 = regression_metrics(y_test, pred)
                model_pkg.append([desc, mae, mse, rmse, r2, lin_cdf])
                
            if model == 'Decision Tree Regressor':
                desc, pred, dt_imp = decision_tree_regression(X_train, X_test, y_train, y_test)
                dt_imp = dt_imp.to_html(classes='data table')
                mae, mse, rmse, r2 = regression_metrics(y_test, pred)
                model_pkg.append([desc, mae, mse, rmse, r2, dt_imp])
                
            if model == 'Random Forest Regressor':
                desc, pred, rf_imp = random_forest_regression(X_train, X_test, y_train, y_test)
                rf_imp = rf_imp.to_html(classes='data table')
                mae, mse, rmse, r2 = regression_metrics(y_test, pred)
                model_pkg.append([desc, mae, mse, rmse, r2, rf_imp])
                
            if model == 'Logistic Regression':
                desc, pred, log_cdf = logistic_regression(X_train, X_test, y_train, y_test)
                log_cdf = log_cdf.to_html(classes='data table')
                accuracy, conf_mat, class_rep = classification_metrics(y_test, pred)
                conf_mat = pd.DataFrame(conf_mat)
                conf_mat = conf_mat.to_html(classes='data table')
                class_rep_df = pd.DataFrame(class_rep).transpose().to_html(classes='data table')
                model_pkg.append([desc, accuracy, conf_mat, class_rep_df, log_cdf])
                
            if model == 'Decision Tree Classifier':
                desc, pred, dtc_imp = decision_tree_classifier(X_train, X_test, y_train, y_test)
                dtc_imp = dtc_imp.to_html(classes='data table')
                accuracy, conf_mat, class_rep = classification_metrics(y_test, pred)
                conf_mat = pd.DataFrame(conf_mat)
                conf_mat = conf_mat.to_html(classes='data table')
                class_rep_df = pd.DataFrame(class_rep).transpose().to_html(classes='data table')
                model_pkg.append([desc, accuracy, conf_mat, class_rep_df, dtc_imp])
                
            if model == 'Random Forest Classifier':
                desc, pred, rfc_imp = random_forest_classifier(X_train, X_test, y_train, y_test)
                rfc_imp = rfc_imp.to_html(classes='data table')
                accuracy, conf_mat, class_rep = classification_metrics(y_test, pred)
                conf_mat = pd.DataFrame(conf_mat)
                conf_mat = conf_mat.to_html(classes='data table')
                class_rep_df = pd.DataFrame(class_rep).transpose().to_html(classes='data table')
                model_pkg.append([desc, accuracy, conf_mat, class_rep_df, rfc_imp])
    
    if 'Linear Regression' in models or 'Decision Tree Regressor' in models \
    or 'Random Forest Regressor' in models:
        return render_template('result.html', train_len=len(X_train), test_len=len(X_test),
                               model_pkg=model_pkg, alert=1
                              )
    else:
        return render_template('result.html', train_len=len(X_train), test_len=len(X_test),
                               model_pkg=model_pkg, alert=0
                              )

@app.route('/buildML', methods=['POST'])
def model_building():
    if request.method == 'POST':
        
        global target_col
        target_col = request.form.getlist('target')[0]  #it will get the list of the target value 
          
        if df[target_col].nunique() > 10:  #checking for classification or regression
            suggested_ml_models = ['Linear Regression', 'Decision Tree Regressor',
                                   'Random Forest Regressor' 
                                  ]
        else:
            suggested_ml_models = ['Logistic Regression', 'Decision Tree Classifier',
                                   'Random Forest Classifier' 
                                  ]
        
        return render_template('MachineLearning.html', models=suggested_ml_models)  #make machineLearning.html file for this rendering


@app.route('/', methods=['POST'])
def upload_file():                                   #this should be in model file try to use data base to store that file and retreve from there
    if request.method == 'POST':

        if 'file' not in request.files:
            return render_template('index.html', alert=0)
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', alert=1)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path_to_file = 'F:/Uploads/' + filename
            # print(path_to_file)
            df = read_file(path_to_file)
            rows, cols, missing_values_sum, desc_stat = dataset_overview(df)
            col_names = df.columns.tolist()
            data = {'Columns': df.isnull().sum().index.values.tolist(),
                    'Missing Values': df.isnull().sum().values.tolist(),
                    'Data Type': df.dtypes.tolist()
                   }
            mv_table = pd.DataFrame(data)
            return render_template('index.html', alert=1, filename=filename, 
                                   rows=rows, cols=cols, col_names=col_names,
                                   mv_table=mv_table.to_html(classes='data table'),
                                   missing_values_sum=missing_values_sum,
                                   desc_table=df.describe().to_html(classes='data table', header='true')
                                  )     
        else:
            return render_template('index.html', alert=0)

@app.route('/') #home Page for initials
def homePage():
    return render_template('index.html')        

if __name__=="__main__":
    app.run(host='0.0.0.0', threaded=True)