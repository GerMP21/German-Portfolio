---
title: "Logistic Regression for Heart Attacks"
type: page
---

## Abstract
This code segment demonstrates the implementation of a logistic regression model for heart attack prediction using the Heart Attack Analysis and Prediction Dataset. 

## Imports
``` python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```

## Load Dataset
``` python
df = pd.read_csv("/kaggle/input/heart-attack-analysis-prediction-dataset/heart.csv")
df
```

![df](/images/heartattack/df.png "df")

``` python
X = df.drop(['output'], axis=1)
y = df['output']
```

## Scaler
``` python
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
```

## Test Train Split
``` python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state = 43)
```

## Model Training
``` python
model = LogisticRegression()
model.fit(X_train,y_train)
```

## Test Model
``` python
y_pred = model.predict(X_test)
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
```

> Accuracy: 0.8852459016393442

## Analysis
``` python
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})


print(feature_importance.sort_values(by='Coefficient', ascending=False))
```

![features](/images/heartattack/features.png "features")

``` python
print(classification_report(y_test,y_pred))
```

![report](/images/heartattack/df.png "report")
