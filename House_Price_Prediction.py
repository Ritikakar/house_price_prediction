
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import matplotlib.pyplot as plt

# this whole code will predict the price of the house of your choice

data = pd.read_csv(r"D:\git demo\HousePrice\training.csv")

#inspect the first few rows
print(data)

#check the column names and datatypes
print(data.info())

#select the relevant features
features = data[['GrLivArea' , 'BedroomAbvGr' , 'FullBath']]
target = data['SalePrice']

print(features)

#check for missing values
print(features.isnull().sum())


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# initialize the model
model = LinearRegression()

#train the model
model.fit(X_train,y_train)

#make predictions on test set
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) 

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# plot actual vs predicted scores
plt.scatter(y_test,y_pred, alpha=0.5)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual Vs Predicted Scores")
plt.show()

# enter the values of your choice to calculate the price for your desired house
liv_area = int(input("enter the desired area for your house"))
bed_area = int(input("enter number of bedrooms"))
bath_area = int(input("enter number of bathrooms"))

new_data = pd.DataFrame({
    'GrLivArea': [liv_area],
    'BedroomAbvGr': [bed_area],
    'FullBath': [bath_area]
})

# Make a prediction using the trained model
predicted_price = model.predict(new_data)

# Print the predicted price
print(f"Predicted House Price: ${predicted_price[0]:,.2f}")









