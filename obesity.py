import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import  accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import  train_test_split,GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.cluster import KMeans
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)

df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

df["bmi"] = df["Weight"] / (df["Height"] ** 2)
df["walking_noFAVC"] = (df["MTRANS"] == "Walking") & (df["FAVC"] == "no")

test_df["bmi"] = test_df["Weight"] / (test_df["Height"] ** 2)
test_df["walking_noFAVC"] = (test_df["MTRANS"] == "Walking") & (test_df["FAVC"] == "no")


new_df = pd.get_dummies(df, columns=["MTRANS",
                                        "Gender",
                                         "family_history_with_overweight", "FAVC", "SMOKE",
                                             "SCC",
                                             "walking_noFAVC",
                                         ], dtype="int64")

new_test_df = pd.get_dummies(test_df, columns=["MTRANS",
                                        "Gender",
                                         "family_history_with_overweight", "FAVC", "SMOKE",
                                             "SCC",
                                             "walking_noFAVC",
                                         ], dtype="int64")


X = new_df.drop(columns=["NObeyesdad", "id"])

y = df["NObeyesdad"]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_test = new_test_df.drop(columns="id")

columns = X_train.columns

"""
No missing values
will scale
"""
# Scaling

# Label encoder has the same functionality as Ordinal ecnoder
# one is only for labels(targets) and the other for features
label_encoder = LabelEncoder()
ordinal_encoder = OrdinalEncoder()
label_encoding_cols = ["CAEC", "CALC"]

# replace two rows of 'always' category to 'sometimes' (the most frequent) that is only found in
# test set and not in training
X_test.loc[X_test[X_test["CALC"] == "Always"].index, "CALC"] = "Sometimes"

X_train[label_encoding_cols] = ordinal_encoder.fit_transform(X_train[label_encoding_cols])
X_val[label_encoding_cols] = ordinal_encoder.transform(X_val[label_encoding_cols])
X_test[label_encoding_cols] = ordinal_encoder.transform(X_test[label_encoding_cols])

y_train = label_encoder.fit_transform(y_train)

y_val = label_encoder.transform(y_val)

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42, max_depth=21, n_estimators=90)
# params = {
#     "max_depth":np.arange(15, 30, 2),
#     "n_estimators":np.arange(85, 101, 5)}
#
# cv = GridSearchCV(model, params, cv=5)

model.fit(X_train, y_train)

preds = model.predict(X_val)

# importances = pd.DataFrame({
#     "features": columns,
#     "weights":model.feature_importances_
# }).sort_values("weights", ascending=False)
#
# print(importances)

print(f"Model accuracy on training data: {accuracy_score(y_train, model.predict(X_train))}")
# print(f"Baseline model accuracy: {accuracy_score(y_val, baseline_model)}")
print(f"Model Accuracy on testing data: {accuracy_score(y_val, preds)}")

test_pred = model.predict(X_test)
test_pred_labels = label_encoder.inverse_transform(test_pred)

answer = pd.DataFrame(
    {
        "id": new_test_df["id"],
        "NObeyesdad": test_pred_labels
    }
)
answer.to_csv("answer.csv", index=False)

