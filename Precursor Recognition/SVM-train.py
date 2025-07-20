import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from ReadDataCNN import getData

txt_dir = r'./data/alldata/'
EMR_feature_data,AE_feature_data,GAS_feature_data,EMR_feature_pad_data,AE_feature_pad_data,GAS_feature_pad_data,EMR_label_data,AE_label_data,GAS_label_data,EMR_feature_data_len,AE_feature_data_len,GAS_feature_data_len = getData(txt_dir)

# 
X_train, X_test, y_train, y_test = train_test_split(AE_feature_pad_data, AE_label_data, test_size=0.1, random_state=42)
X_train_da, _, y_train_da, _ = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
kernels = ["linear", "poly", "rbf", "sigmoid"]
results = []

for kernel  in kernels:
    results = []
    model = SVC(kernel=kernel ,C=1e5)
    model.fit(X_train_da, y_train_da)
    y_proba = model.predict(AE_feature_pad_data)
    
    for i in range(len(AE_label_data)):
        results.append([y_proba[i], AE_label_data[i]])
    df = pd.DataFrame(results, columns=["predicted_proba", "true_label"])
    df.to_excel("./tmpSVMAE/{}_svm_results.xlsx".format(kernel), index=False)

