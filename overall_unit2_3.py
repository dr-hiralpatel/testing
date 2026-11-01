import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------------------------------------
# Step 1 : Load Dataset
df = pd.read_csv("student_data.csv")
print("Complete Dataset")
print(df)

# --------------------------------------------------
# Step 2 : Understand Dataset
print("\nShape :", df.shape)
print("\nColumns :", df.columns.tolist())

# --------------------------------------------------
# Step 3 : Identify Features and Label
X = df[["StudyHours"]]      # Feature
y = df["Marks"]             # Label

print("\nFeature (Input)")
print(X.head())

print("\nLabel (Output)")
print(y.head())

# --------------------------------------------------
# Step 4 : Type of Machine Learning
print("\nMachine Learning Type : Supervised Learning")
print("Problem Type(Regression / Classification): Regression")

# --------------------------------------------------
# Step 5 : Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)
print("\nTraining Records :", len(X_train))
print("Testing Records  :", len(X_test))

# --------------------------------------------------
# Step 6 : Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# Step 7 : Prediction
y_pred = model.predict(X_test)

print("\nActual Marks")
print(y_test.values)
print("\nPredicted Marks")
print(y_pred)

# --------------------------------------------------
# Step 8 : Error Calculation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nMean Absolute Error (MAE) :", mae)
print("Mean Squared Error (MSE) :", mse)

# --------------------------------------------------
# Step 9 : Predict New Student
hours = [[6]]
prediction = model.predict(hours)

print("\nIf student studies 6 hours then Predicted Marks :", prediction[0])