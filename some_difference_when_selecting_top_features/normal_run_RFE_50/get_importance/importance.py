import joblib
import os,sys
# sys.path.append(r"C:\Users\Jerry Wang\Desktop\Paper_final_collection\lmax\test\get_importance")
import pandas as pd
import numpy as np
model_file = r"total_model_RFE50_kfold10_LGBMRegressor_225.model"
input_df = pd.read_csv(r"train0.8.csv")
feature_df = input_df.iloc[:, :-1]
print(feature_df.shape)


model = joblib.load(model_file)

var_filter = model.variance_filter

feature_df_var = feature_df.loc[:,var_filter.get_support()]
print(feature_df_var.shape)

feature_df_sele = feature_df_var.loc[:,model.selector.get_support()]
print(feature_df_sele.shape)

importances = []
for m in model.models:
    importances.append(m.feature_importances_)
print(len(importances))
print(importances[5])

df_importance = pd.DataFrame([feature_df_sele.columns, importances[5]], index=["feature_name", "importance"]).T
print(df_importance)
df_importance.sort_values(by="importance", ascending=False, inplace=True)
print(df_importance)
df_importance.to_csv('importance.csv')