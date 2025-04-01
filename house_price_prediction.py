# Artificial-Intelligence-DA
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Generate synthetic data for house sizes (in square feet) and prices (in $1000s)
np.random.seed(42)  # For reproducibility
house_sizes = np.random.randint(800, 3500, 100)  # Random house sizes between 800 and 3500 sq ft
prices = house_sizes * 0.3 + np.random.normal(0, 20, 100)  # Price = 0.3 * size + some noise

# Reshape the data for sklearn (it expects 2D arrays)
X = house_sizes.reshape(-1, 1)  # Features (house sizes)
y = prices  # Target (prices)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# Step 6: Print the slope (coefficient) and intercept of the line
print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Step 7: Visualize the results
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_test, y_pred, color='red', label='Linear Regression Line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($1000s)')
plt.title('House Price Prediction Using Linear Regression')
plt.legend()
plt.show()

# Step 8: Predict the price of a new house
new_house_size = np.array([[2000]])  # Example: 2000 sq ft house
predicted_price = model.predict(new_house_size)
print(f"Predicted price for a 2000 sq ft house: ${predicted_price[0]:.2f}k")