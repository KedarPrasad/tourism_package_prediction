import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
# Make sure the model filename and repo_id are correct
model_path = hf_hub_download(repo_id="kpiitkgp/tourism_package_prediction", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the Wellness Tourism Package based on their details and interaction data.
Please enter the customer details and interaction data below to get a prediction.
""")

# User input - Customer Details
st.header("Customer Details")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
preferredpropertystar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
designation = st.selectbox("Designation", ["Manager","Executive","Senior Manager","AVP","VP"]) 
monthlyincome = st.number_input("Monthly Income", min_value=0, value=20000)


# User input - Customer Interaction Data
st.header("Customer Interaction Data")
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=100, value=10)


# Assemble input into DataFrame
# Ensure column names match the training data and are in the correct order
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': citytier,
    'DurationOfPitch': durationofpitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': maritalstatus,
    'NumberOfTrips': numberoftrips,
    'Passport': passport,
    'PitchSatisfactionScore': pitchsatisfactionscore,
    'OwnCar': owncar,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'Designation': designation,
    'MonthlyIncome': monthlyincome,
    'ProductPitched': productpitched,
    'NumberOfFollowups': numberoffollowups,
}])


if st.button("Predict Purchase"):
    # Ensure the column order matches the training data before making a prediction
    # This is crucial if your model pipeline doesn't handle column order internally
    # Get the column order from the training data (assuming Xtrain is available or you know the order)
    # For example: expected_columns = ['Age', 'TypeofContact', ..., 'NumberOfFollowups'] # Replace with actual order
    # input_data = input_data[expected_columns] # Reorder columns

    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= 0.45).astype(int) # Using the classification threshold from training

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("The model predicts that the customer **will likely purchase** the Wellness Tourism Package.")
    else:
        st.info("The model predicts that the customer **will likely not purchase** the Wellness Tourism Package.")

    st.write(f"Predicted probability of purchase: {prediction_proba[0]:.4f}")
