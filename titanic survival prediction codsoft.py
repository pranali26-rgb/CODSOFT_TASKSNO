import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os

file_path = os.path.join(os.path.dirname(__file__), "Titanic-Dataset.csv")
df = pd.read_csv(file_path)

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Survival distribution

plt.figure(figsize=(6,4))

sns.countplot(
    x="Survived",
    data=df
)

plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.show()


# Survival by gender

plt.figure(figsize=(6,4))

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.show()


# Survival by passenger class

plt.figure(figsize=(6,4))

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=df
)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.show()


# Age distribution

plt.figure(figsize=(7,4))

sns.histplot(
    df["Age"],
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()


data = df.copy()

columns_to_drop = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]

data = data.drop(
    columns=[
        col for col in columns_to_drop
        if col in data.columns
    ]
)


if "Age" in data.columns:

    data["Age"] = data["Age"].fillna(
        data["Age"].median()
    )


if "Embarked" in data.columns:

    data["Embarked"] = data["Embarked"].fillna(
        data["Embarked"].mode()[0]
    )


if "Fare" in data.columns:

    data["Fare"] = data["Fare"].fillna(
        data["Fare"].median()
    )


if "Sex" in data.columns:

    data["Sex"] = data["Sex"].map({
        "male": 0,
        "female": 1
    })


if "Embarked" in data.columns:

    data["Embarked"] = data["Embarked"].map({
        "S": 0,
        "C": 1,
        "Q": 2
    })


print("\nProcessed Dataset:")
print(data.head())

print("\nMissing Values After Preprocessing:")
print(data.isnull().sum())

X = data.drop(
    "Survived",
    axis=1
)

y = data["Survived"]


print("\nFeatures Used:")
print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nModel training completed!")


y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================================")
print("MODEL RESULTS")
print("================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        model.feature_importances_

})


importance = importance.sort_values(

    by="Importance",

    ascending=False
)


print("\nFeature Importance:")

print(importance)


plt.figure(figsize=(8,5))

sns.barplot(

    x="Importance",

    y="Feature",

    data=importance
)

plt.title("Feature Importance")

plt.show()


new_passenger = pd.DataFrame({

    "Pclass": [3],

    "Sex": [1],

    "Age": [22],

    "SibSp": [1],

    "Parch": [0],

    "Fare": [7.25],

    "Embarked": [0]

})


prediction = model.predict(
    new_passenger
)


print("\n================================")
print("NEW PASSENGER PREDICTION")
print("================================")


if prediction[0] == 1:

    print("Prediction: Passenger Survived")

else:

    print("Prediction: Passenger Did Not Survive")