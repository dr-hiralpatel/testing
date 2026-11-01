import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("student.csv")
print(df.head())
'''
#Features & Labels
X = df[['Study_Hours', 'Attendance', 'Assignments_Score', 'Previous_Score']]
#y_class = df['Result']        # Classification
y_reg = df['Final_Marks']     # Regression

#Categorical Data Conversion
df = pd.get_dummies(df, columns=['Participation', 'Internet_Usage'], drop_first=True)
print(df)


#Training & Testing Set Creation
#from sklearn.model_selection import train_test_split
X = df.drop(['Student_ID','Result','Final_Marks'], axis=1)
y_class = df['Result']
y_reg = df['Final_Marks']

#X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)


Classification Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

model_c = DecisionTreeClassifier()
model_c.fit(X_train, y_train_c)

y_pred_c = model_c.predict(X_test)

print("Accuracy:", accuracy_score(y_test_c, y_pred_c))

#Regression Model design
#from sklearn.linear_model import LinearRegression

model_r = LinearRegression()
model_r.fit(X_train, y_train)

y_pred = model_r.predict(X_test)
print(X_test)
print(y_pred)


#MSE calculation
#from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
plt.figure()
#plt.plot(range(len(y_test)), y_test.values, marker='o', label='Actual')
#plt.plot(range(len(y_pred)), y_pred, marker='x', label='Predicted')
plt.plot(X_test.values, y_test.values,marker='o')
plt.plot(X_test.values, y_pred,marker='<')
plt.show()

#MAE calculation
#from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

#Overfitting & Underfitting
train_acc = model_c.score(X_train, y_train_c)
test_acc = model_c.score(X_test, y_test_c)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

'''


