# House Price Prediction Using Linear Regression

This project demonstrates the use of **linear regression** for predicting house prices based on house sizes (in square feet). It uses synthetic data to model the relationship between house size and price, visualizes the results, and evaluates the performance of the model using the mean squared error (MSE).

## Steps Involved

1. **Generate Synthetic Data**: 
   - Random house sizes (in square feet) are generated within a range of 800 to 3500 sq ft.
   - Prices are generated using the formula: `Price = 0.3 * size + noise`, where `noise` is added to introduce some randomness to the data.

2. **Train-Test Split**:
   - The dataset is split into training (80%) and testing (20%) sets using `train_test_split`.

3. **Linear Regression**:
   - A linear regression model is created and trained on the training data.
   - The model makes predictions on the test data.

4. **Model Evaluation**:
   - The model’s performance is evaluated using **Mean Squared Error (MSE)**.

5. **Visualization**:
   - The training and test data points are visualized, along with the fitted linear regression line.

6. **Prediction**:
   - A prediction is made for a new house with a given size (e.g., 2000 sq ft).

## Requirements

You need the following Python libraries to run the code:
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install them using pip:

```bash
pip install numpy matplotlib scikit-learn