import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sklearn as skl

df= pd.read_csv("C:/Users/Pranali/OneDrive/Desktop/TASK 3/archive (3)/IRIS.csv")
print(df.head())
print(df.columns.tolist())
print(df.info())
print(df.isnull().sum())

print(df['species'].value_counts())
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.shape)

X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y=df['species']
print("Input features\n :")
print(X.head())
print("Target output ")
print(y.head())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y=encoder.fit_transform(y)
print("\nEncoded species:")
print(y[:10])

print("\nSpecies mapping:")
for number, species in enumerate(encoder.classes_):
    print(number, "=", species)
    
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state =42,
    stratify=y)

print("\nTraining Data:")
print(X_train.shape)

print("Testing Data :")
print(X_test.shape)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100,
    random_state = 42)
 
model.fit(X_train,y_train)
print("Model Training Completed")

y_pred = model.predict(X_test)
print("Actual values:")
print(y_test)

print("Predicted values : ")
print(y_pred)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test , y_pred)
print("model accuracy : ")
print(accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
print("Confusion matrix : ")
print(cm)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

# New flower measurements
new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict species
prediction = model.predict(new_flower)

# Convert number back to species name
predicted_species = encoder.inverse_transform(prediction)

print("\nNew Flower Prediction:")
print("Predicted Species:", predicted_species[0])


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=encoder.classes_
)

plt.title("Iris Flower Classification - Confusion Matrix")
plt.show()