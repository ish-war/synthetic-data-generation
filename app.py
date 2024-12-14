
import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load the saved models and scaler
with open("random_forest_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

nn_model = load_model("neural_network_model_tuned.h5")

with open("scaler_tuned.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit application
def main():
    st.title("Nurse Hourly Pay Rate Predictor")

    # User Inputs
    job_title = st.selectbox("Select Job Title", [
        "RegisteredNurse_ICU", "RegisteredNurse_MedSurg", "RegisteredNurse_Telemetry",
        "RegisteredNurse_Oncology", "RegisteredNurse_Pediatric", "PhysioTherapist",
        "LabTechnician", "RegisteredNurse_CriticalCare", "RegisteredNurse_Cardiology",
        "RegisteredNurse_Surgery"
    ])

    location = st.selectbox("Select Location", [
        "Dallas, TX", "Atlanta, GA", "New York, NY", "Philadelphia, PA", "Washington, DC",
        "San Francisco, CA", "Los Angeles, CA", "Seattle, WA", "Chicago, IL", "San Diego, CA",
        "Miami, FL", "Boston, MA", "Detroit, MI", "Phoenix, AZ", "Houston, TX"
    ])

    hospital_name = st.text_input("Enter Hospital Name", "Example Hospital")
    contract_start_date = st.date_input("Select Contract Start Date")
    contract_end_date = st.date_input("Select Contract End Date")

    model_choice = st.radio("Select Model for Prediction", ("Random Forest", "Neural Network"))

    if st.button("Predict Hourly Rate"):
        # Prepare input data
        user_data = pd.DataFrame({
            'Job Title': [job_title],
            'Location': [location],
            'Hospital Name': [hospital_name],
            'Contract Start Date': [contract_start_date],
            'Contract End Date': [contract_end_date]
        })

        # Mock City Desirability data (update with actual logic if needed)
        city_factors = {
            'Dallas': 3, 'Atlanta': 4, 'New York': 5, 'Philadelphia': 3, 'Washington': 4,
            'San Francisco': 5, 'Los Angeles': 4, 'Seattle': 5, 'Chicago': 3, 'San Diego': 4,
            'Miami': 4, 'Boston': 5, 'Detroit': 2, 'Phoenix': 3, 'Houston': 3
        }
        user_data['City Desirability'] = user_data['Location'].apply(lambda loc: city_factors.get(loc.split(',')[0], 3))

        # One-hot encode and align columns
        user_data_encoded = pd.get_dummies(user_data[['Job Title', 'Location', 'City Desirability']], drop_first=True)
        user_data_encoded = user_data_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # Scale data for neural network
        user_data_scaled = scaler.transform(user_data_encoded)

        # Prediction
        if model_choice == "Random Forest":
            prediction = rf_model.predict(user_data_encoded)[0]
        else:
            prediction = nn_model.predict(user_data_scaled)[0][0]

        # Display result
        st.success(f"Predicted Hourly Pay Rate: ${prediction:.2f}")

if __name__ == "__main__":
    main()
