import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('C:/Users/shahe/Desktop/ML DEPLOYMENT/trained_model.sav', 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return "The person is diabetic" if prediction[0] else "The person is not diabetic"

# Main function for the Streamlit app
def main():
    # Title
    st.title('Diabetes Prediction Web App')

    # Input fields organized in two columns
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
        Glucose = st.number_input('Glucose Level', min_value=0)
        BloodPressure = st.number_input('Blood Pressure value', min_value=0)
        SkinThickness = st.number_input('Skin Thickness value', min_value=0)
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0)
        BMI = st.number_input('BMI value', min_value=0.0)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0)
        Age = st.number_input('Age of the Person', min_value=0)

    # Button for prediction
    if st.button('Diabetes Test Result'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diagnosis = diabetes_prediction(input_data)
        st.success(diagnosis)

if __name__ == '__main__':
    main()
