# Music-Popularity-Forecasting-Pipeline
Predict song popularity using audio features and a Random Forest model. Full ML pipeline includes data cleaning, scaling, training/testing, evaluation, feature importance analysis, and visualization to understand what drives music success.


Song Popularity Prediction
Predict song popularity using audio features with a Random Forest Regressor.
Project Overview
This project predicts the popularity of songs (0–100 scale) based on features like danceability, energy, valence, tempo, and duration.
Goals:
1.	Predict song popularity using machine learning.
2.	Identify which audio features influence popularity most.
3.	Visualize model predictions against actual popularity.
Dataset
•	Entries: 114,000 songs
•	Key Features:
o	danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms
•	Target Variable: popularity (0–100)
* Only numeric audio features were used for prediction. *
Data Preprocessing
•	Missing values replaced with column means.
•	Features standardized with StandardScaler (mean=0, std=1).
•	Dataset split:
o	Training: 80%
o	Testing: 20%
Model
•	Algorithm: Random Forest Regressor
•	Parameters:
o	n_estimators=100 → number of trees
o	random_state=42 → reproducible results
o	n_jobs=-1 → uses all CPU cores
•	Model trained on X_train and y_train, tested on X_test.





Evaluation Metrics
•	R² (R-squared): How much variance in popularity is explained by the model.
•	RMSE (Root Mean Squared Error): Average prediction error in popularity points.
Example Output:
R²: 0.55
RMSE: 14.2
Feature Importance
Random Forest allows us to see which features impact predictions the most.
Top 5 Features Example:
danceability: 0.22
energy: 0.18
valence: 0.15
tempo: 0.12
loudness: 0.10
•	Insight: Danceability, energy, and valence are the strongest predictors of song popularity.
Predicted vs Actual Popularity
•	Scatter plot compares predictions with actual values.
•	Diagonal line represents perfect predictions (y=x).
•	Points close to the line indicate accurate predictions.
Conclusion
•	The Random Forest model successfully predicts song popularity using audio features.
•	Features like danceability, energy, and valence have the largest influence.
•	Model performance is moderate (R² ≈ 0.55), indicating other factors like social trends, marketing, or artist popularity also affect song success.
•	Visualizations provide insight into model accuracy and feature importance.

<img width="468" height="647" alt="image" src="https://github.com/user-attachments/assets/ddfa751f-c5d5-4131-80ac-cef877fe9429" />
