

###### Song Popularity Prediction Pipeline



# Used for loading, cleaning, and manipulating datasets (DataFrames)
import pandas as pd  

# Used for numerical operations (arrays, math functions, square root, etc.)
import numpy as np  

# Used to create graphs and visualizations (plots, charts)
import matplotlib.pyplot as plt  

# StandardScaler: scales features so they have mean = 0 and standard deviation = 1
from sklearn.preprocessing import StandardScaler  

# train_test_split: Splits dataset into training and testing sets
from sklearn.model_selection import train_test_split  

# RandomForestRegressor: Machine learning model that predicts numerical values
from sklearn.ensemble import RandomForestRegressor  

# mean_squared_error: Measures prediction error
from sklearn.metrics import mean_squared_error, r2_score  


# Loading dataset 
data = pd.read_csv("dataset.csv")


# Printing data set information
print("First 5 rows of dataset:")
print(data.head())

print("\nDataset info:")
print(data.info())



# Selecting features and target
# Description of all of the table information 
features = [
    'danceability','energy','loudness','speechiness','acousticness',
    'instrumentalness','liveness','valence','tempo','duration_ms'
]


# Initializing variable with table sections 
X = data[features]


# Initializing variable with popularity section
y = data['popularity']




# calculating the mean average of each column
X = X.fillna(X.mean())


# standarizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




# Split the dataset into training (80%) and testing (20%) sets
# Training data is used to teach the model, testing data evaluates performance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)



# Create a Random Forest model to predict song popularity
# n_estimators=100 → number of trees in the forest
# random_state=42 → ensures reproducible results
# n_jobs=-1 → uses all CPU cores for faster training

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1   # Uses all CPU cores (faster)
)


# Train the Random Forest model on the training data
# The model learns patterns between features (X_train) and popularity (y_train)
model.fit(X_train, y_train)



# Use the trained model to predict popularity for the test set
y_pred = model.predict(X_test)




# Evaluate the model:
# R² measures how well predictions explain variance
# RMSE measures the average prediction error

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 

print("\nModel Evaluation:")
print("R²:", r2)
print("RMSE:", rmse)




# Analyze which features most influence song popularity, feature importance scores from Random Forest
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importance for Song Popularity")
plt.bar(range(len(features)), importances[indices])
plt.xticks(range(len(features)),
           [features[i] for i in indices],
           rotation=45)
plt.tight_layout()
plt.show()

print("\nTop 5 Features Influencing Popularity:")
for i in indices[:5]:
    print(f"{features[i]}: {importances[i]:.4f}")


# Plot predicted vs actual popularity
# Each point is a song; red line = perfect prediction
# Helps visualize how well the model predicts unseen data
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Predicted vs Actual Song Popularity")
plt.plot([0,100], [0,100])  # reference line
plt.tight_layout()
plt.show()