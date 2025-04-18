import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
import pickle

def multioutput_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracies = []
    for i in range(y_true.shape[1]):
        accuracies.append(accuracy_score(y_true[:, i], y_pred[:, i]))
    return np.mean(accuracies)


multi_accuracy = make_scorer(multioutput_accuracy)


df_train = pd.read_csv("train_data.csv")
df_train["sample"] = df_train["sample"].fillna("")

X = df_train["sample"]
y = df_train[["dialect", "category"]]


vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=10000)
X_vect = vectorizer.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(
    X_vect, y, test_size=0.2, random_state=42
)


lr = LogisticRegression(max_iter=1000, random_state=42)


param_grid = {'estimator__C': [0.1, 1, 10]}
grid = GridSearchCV(MultiOutputClassifier(lr), param_grid, cv=3, scoring=multi_accuracy)
grid.fit(X_train, y_train)


clf = grid.best_estimator_
print("Cei mai buni parametri:", grid.best_params_)


y_val_pred = clf.predict(X_val)
acc_dialect = accuracy_score(y_val["dialect"], y_val_pred[:, 0])
acc_category = accuracy_score(y_val["category"], y_val_pred[:, 1])
print("Acuratețe dialect (validare):", acc_dialect)
print("Acuratețe category (validare):", acc_category)


with open(r"C:\Users\alex2\Desktop\model.pkl", "wb") as f:
    pickle.dump((clf, vectorizer), f)


df_test = pd.read_csv("test-data.csv")
df_test["sample"] = df_test["sample"].fillna("")

X_test = df_test["sample"]
X_test_vect = vectorizer.transform(X_test)


y_test_pred = clf.predict(X_test_vect)
y_test_pred = np.nan_to_num(y_test_pred, nan=0)

if "datapoint" in df_test.columns:
    id_series = df_test["datapoint"].fillna(0)
elif "datapointID" in df_test.columns:
    id_series = df_test["datapointID"].fillna(0)
else:
    id_series = df_test.index

results = pd.DataFrame({
    "datapoint": id_series,
    "dialect": y_test_pred[:, 0],
    "category": y_test_pred[:, 1]
})

output_csv = r"C:\Users\alex2\Desktop\prediction_results.csv"
results.to_csv(output_csv, index=False)
print(f"Fișierul cu predicții a fost salvat la: {output_csv}")
