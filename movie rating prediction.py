import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


file_path = r"C:/Users/Pranali/OneDrive/Desktop/movie prediction task 2/archive (3)/IMDb Movies India.csv"

df = pd.read_csv(file_path, encoding='latin1')

print("Dataset loaded successfully!")

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)

df.columns = df.columns.str.strip()

df['Year'] = (
    df['Year']
    .astype(str)
    .str.extract(r'(\d{4})')[0]
)

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

df['Duration'] = (
    df['Duration']
    .astype(str)
    .str.extract(r'(\d+)')[0]
)

df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

df['Votes'] = (
    df['Votes']
    .astype(str)
    .str.replace(',', '', regex=False)
)

df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')


df = df.dropna(subset=['Rating'])

print("\nRows after removing movies without Rating:")
print(df.shape)
========================================================

# Numerical columns
df['Year'] = df['Year'].fillna(df['Year'].median())
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())


# Categorical columns
categorical_columns = [
    'Genre',
    'Director',
    'Actor 1',
    'Actor 2',
    'Actor 3'
]

for col in categorical_columns:
    df[col] = df[col].fillna('Unknown')


print("\nMissing values after cleaning:")
print(df.isnull().sum())


features = [
    'Year',
    'Duration',
    'Votes',
    'Genre',
    'Director',
    'Actor 1',
    'Actor 2',
    'Actor 3'
]

X = df[features]

y = df['Rating']

numerical_features = [
    'Year',
    'Duration',
    'Votes'
]

categorical_features = [
    'Genre',
    'Director',
    'Actor 1',
    'Actor 2',
    'Actor 3'
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(
                handle_unknown='ignore'
            ),
            categorical_features
        ),
        (
            'numerical',
            'passthrough',
            numerical_features
        )
    ]
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))

pipeline.fit(X_train, y_train)

print("\nModel training completed successfully!")


y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("MAE  :", round(mae, 3))
print("MSE  :", round(mse, 3))
print("RMSE :", round(rmse, 3))
print("R²   :", round(r2, 3))


plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual IMDb Rating")
plt.ylabel("Predicted IMDb Rating")

plt.title("Actual vs Predicted IMDb Ratings")

plt.show()


sample_movie = pd.DataFrame({
    'Year': [2024],
    'Duration': [130],
    'Votes': [50000],
    'Genre': ['Action, Drama'],
    'Director': ['Rajkumar Hirani'],
    'Actor 1': ['Aamir Khan'],
    'Actor 2': ['Kareena Kapoor'],
    'Actor 3': ['Unknown']
})


predicted_rating = pipeline.predict(sample_movie)


print("\n================================")
print("SAMPLE MOVIE PREDICTION")
print("================================")

print("Predicted IMDb Rating:",
      round(predicted_rating[0], 2))


sample_movie2 = pd.DataFrame({
    'Year': [2020],
    'Duration': [120],
    'Votes': [10000],
    'Genre': ['Comedy'],
    'Director': ['Unknown'],
    'Actor 1': ['Unknown'],
    'Actor 2': ['Unknown'],
    'Actor 3': ['Unknown']
})

predicted_rating2 = pipeline.predict(sample_movie2)

print("\nSecond sample prediction:",
      round(predicted_rating2[0], 2))
