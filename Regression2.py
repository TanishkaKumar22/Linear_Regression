import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer  # NEW: Import SimpleImputer for handling NaNs

file_path = r"C:\Users\kumar\Downloads\archive (2)\Food_Delivery_Times.csv"
df = pd.read_csv(file_path)

#for label encoding
# here fit_transform first learns that high=2,medium=1,low=0, this is called "mapping the features"
#fit transform always returns a new array hence no inplace return true
label_encoder=LabelEncoder()
df['Traffic_Level']=label_encoder.fit_transform(df['Traffic_Level'])

#for one hot encoding
#converts categorical columns into one-hot encoded columns
# get_dummies creates dummy variables for each category in dataframe df , this is then replaced by binary columns
df=pd.get_dummies(df,columns=['Weather','Time_of_Day','Vehicle_Type'])

#we will now drop 'order_id' as it is not useful for regression
#inplace=true means that will change the table with new values rather than returning new table
df.drop(columns=['Order_ID'], inplace=True)

#defining feature value and target value
x=df.drop(columns=['Delivery_Time_min'])
y=df['Delivery_Time_min']

#handling missing values by replacing NaNs with mean values
imputer = SimpleImputer(strategy="mean")  # NEW: Replace missing values with column mean
x = imputer.fit_transform(x)  # Apply imputation

#split the dataset into training and testing sets:
#so train_test_split divides the data in x_train , x_test, y_train,y_test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#train linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

#prediction
#model makes predictions based on x_test , tested set
y_pred=model.predict(X_test)

#so y test is the actual values and y pred is the ones we got
plt.plot(Y_test.values, label="Actual Delivery Time", linestyle="-", marker="o", color="blue")
plt.plot(y_pred, label="Predicted Delivery Time", linestyle="-", marker="x", color="red")

plt.xlabel("Index")
plt.ylabel("Delivery Time")
plt.title("Actual vs Predicted Delivery Time")

#is actually used to show the labels , after we have assigned them
plt.legend()

#adds grid to the plot
plt.grid(True)

plt.show()

#mae: Mean Absolute Error
# measures the average absolute difference between the predicted and actual values
mae = mean_absolute_error(Y_test, y_pred)

#MSE penalizes larger errors more than MAE by squaring the differences between the predicted and actual values.
mse = mean_squared_error(Y_test, y_pred)

#R² tells you how well the model explains the variation in the dependent variable.
# It ranges from 0 to 1, where 1 means the model perfectly fits the data and 0 means the model doesn't explain any of the variance.
r2 = r2_score(Y_test, y_pred)

# Print the efficiency metrics
print(f"Mean Absolute Error : {mae}")
print(f"Mean Squared Error : {mse}")
print(f"R-squared: {r2}")

#any number of rows will be selected
