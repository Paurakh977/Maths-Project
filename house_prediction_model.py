import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(0)
house_sizes = np.random.randint(800, 4000, 500).reshape(-1, 1)  
prices = 50 * house_sizes + np.random.normal(0, 10000, 500).reshape(-1, 1)  
X_train, X_test, y_train, y_test = train_test_split(house_sizes, prices, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predicted_price_2500 = model.predict(np.array([[2500]]))
print(f"Predicted price for a house of size 2500 sq. ft.: ${predicted_price_2500[0][0]:.2f}")
actual_price_2500 = 50 * 2500  
error_2500 = abs(predicted_price_2500[0][0] - actual_price_2500)
print(f"Error in prediction for a house of size 2500 sq. ft.: ${error_2500:.2f}")
plt.scatter(X_test, y_test, color='#095ddb', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.scatter(2500, predicted_price_2500[0][0], color='black', label='Predicted Price for 2500 sq. ft.', marker='x', s=200, linewidths=2)
plt.xlabel('House Size (sq. ft.)')
plt.ylabel('House Price ($)')
plt.title('House Price Prediction')
plt.text(2400, 180000, f'Predicted Price: ${predicted_price_2500[0][0]:.2f}', fontsize=18, ha='right', va='bottom')
plt.text(2600, 180000, f'Error: ${error_2500:.2f}', fontsize=18, ha='left', va='top')
plt.legend()
plt.grid(True)
plt.show()
