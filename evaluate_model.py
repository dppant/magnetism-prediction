# import libraries
import pandas as pd
import pickle as pkl
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from tensorflow.keras.models import load_model


# load input data and separate features, target variable
test = pd.read_csv('data/train_test/test.csv', header = 0)
y_test = test['M']
X_test = test.drop(columns = ['M'])


# load standard scaler
scaler = pkl.load(open('models/standard_scaler/scaler.pkl', 'rb'))

# transform input data using saved standard scaler
cols_to_scale = ['Fe','S1','S2','S3','S4','Ni','Co','Cr','Mn','Se','S','Te'] # list for cols to scale
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale]) # scale test data


# load base models
ANN_model = load_model("models/base_models/ann.h5")
svr = pkl.load(open('models/base_models/svr.pkl', 'rb'))
rf_reg = pkl.load(open('models/base_models/rf_reg.pkl', 'rb'))
knn_reg = pkl.load(open('models/base_models/knn_reg.pkl', 'rb'))
xgb_reg = pkl.load(open('models/base_models/xgb_reg.pkl', 'rb'))
dt_reg = pkl.load(open('models/base_models/dt_reg.pkl', 'rb'))

# load meta model
rf_meta_final = pkl.load(open('models/meta_model/rf_meta_final.pkl', 'rb'))

# prepare new features based on base classifiers for test set
y_pred_test_ann = ANN_model.predict(X_test)
y_pred_test_ann = y_pred_test_ann.reshape(y_pred_test_ann.shape[0],)
y_pred_test_svm = svr.predict(X_test)
y_pred_test_rf = rf_reg.predict(X_test)
y_pred_test_knn = knn_reg.predict(X_test)
y_pred_test_xgb = xgb_reg.predict(X_test)
y_pred_test_dt = dt_reg.predict(X_test)

# get new features from base models
X_test_stacking = pd.DataFrame()
X_test_stacking['ann'] = y_pred_test_ann
X_test_stacking['svm'] = y_pred_test_svm
X_test_stacking['rf'] = y_pred_test_rf
X_test_stacking['knn'] = y_pred_test_knn
X_test_stacking['xgb'] = y_pred_test_xgb
X_test_stacking['dt'] = y_pred_test_dt

# apply new features to meta classifier
final_prediction = rf_meta_final.predict(X_test_stacking)

# evaluate 

mse = mean_squared_error(final_prediction, y_test)
r2 = r2_score(final_prediction, y_test)
mae = mean_absolute_error(final_prediction, y_test)
# print("MSE = %.2f, R2 Score = %f, MAE=%f"%mse, mae, r2)

print("MSE = {:.2f}, MAE = {:.2f}, R2 Score = {:.2f}".format(mse, mae,r2))
#pd.DataFrame(final_prediction).to_csv("Latest_data_prediction")
