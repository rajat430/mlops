import pandas as pd
import joblib

# -----------------------------------
# Load trained model
# -----------------------------------
model = joblib.load("models/model.pkl")

# -----------------------------------
# Create passenger input
# -----------------------------------
input_data = pd.DataFrame({
    "Pclass": [3],
    "Sex": [1],          # male=1, female=0
    "Age": [22],
    "Fare": [7.25],
  })
# Match training datatype
input_data = input_data.astype("float64")

# -----------------------------------
# Predict
# -----------------------------------
prediction = model.predict(input_data)

# -----------------------------------
# Output
# -----------------------------------
if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")