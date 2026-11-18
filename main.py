
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Student data
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [20, 25, 35, 40, 50, 60, 65, 75, 85, 95]
}

df = pd.DataFrame(data)

# Input and output
X = df[["Study_Hours"]]
y = df["Marks"]

# Train the Machine Learning model
model = LinearRegression()
model.fit(X, y)

# Predict marks for a new student
hours = float(input("Enter daily study hours: "))
prediction = model.predict(pd.DataFrame({
    "Study_Hours": [hours]
}))

print(f"Predicted marks: {prediction[0]:.2f} out of 100")

# Display graph
plt.scatter(df["Study_Hours"], df["Marks"], label="Student Data")
plt.plot(df["Study_Hours"], model.predict(X), label="Prediction Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Performance Prediction")
plt.legend()
plt.show()