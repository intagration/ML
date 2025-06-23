import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Dataset (Years of Experience vs Salary)
# X = Independent variable (Years of Experience)
# y = Dependent variable (Salary in thousands)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000])

# Create Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predicting for new values (optional)
X_test = np.array([[11]])
y_pred = model.predict(X_test)

# Print Coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print(f"Predicted salary for {X_test[0][0]} years experience: ₹{y_pred[0]:.2f}")

# Plotting the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
