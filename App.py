import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the saved scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('project_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up the Streamlit app page configuration
st.set_page_config(page_title="Student Performance Prediction", page_icon="📊", layout="centered")

st.title("🎓 Student Performance Prediction App")
st.markdown("""
    Use this app to predict whether a student will **pass or fail** based on five important features:
    **Free time**, **Going out with friends**, **Age**, **Past class failures**, and **Absences**. 
    Adjust the values below and click **Predict** to see the results.
""")

# Input fields for the top 5 features (no sidebar)
st.header("Input Features")

# Feature 1: Free Time (How much free time after school)
freetime = st.slider('Free Time after School (1: very low, 5: very high)', min_value=1, max_value=5, value=3)

# Feature 2: Go Out (Going out with friends)
goout = st.slider('Go Out with Friends (1: very low, 5: very high)', min_value=1, max_value=5, value=3)

# Feature 3: Age
age = st.slider('Age', min_value=15, max_value=22, value=18)

# Feature 4: Failures (Number of past class failures)
failures = st.slider('Past Class Failures', min_value=0, max_value=3, value=0)

# Feature 5: Absences
absences = st.number_input('Number of Absences', min_value=0, max_value=100, value=4)

# Input data formatting
input_data = np.array([[freetime, goout, age, failures, absences]])

# Define prediction function
def predict():
    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)[0]
    
    return prediction[0], prediction_proba

# Prediction button
if st.button('Predict'):
    # Get prediction and probability
    prediction, prediction_proba = predict()

    # Display results based on prediction
    st.subheader('Prediction Results')

    if prediction == 1:
        st.success('🎉 The model predicts: **PASS**')
    else:
        st.error('🚨 The model predicts: **FAIL**')

    # Display prediction probability
    st.write(f"**Probability of passing**: {prediction_proba[1] * 100:.2f}%")
    st.write(f"**Probability of failing**: {prediction_proba[0] * 100:.2f}%")

    # Circular visualization of prediction probability (Pie chart)
    st.subheader('Prediction Probability (Pie Chart)')

    fig, ax = plt.subplots()
    labels = ['Fail', 'Pass']
    sizes = [prediction_proba[0], prediction_proba[1]]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)  # explode the first slice (Fail)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

# Footer
st.markdown("""
    ---
    **Note**: This prediction is based on a machine learning model trained on the UCI Student Performance dataset, 
    which takes into account factors such as free time, going out with friends, age, failures, and absences to estimate the likelihood of passing or failing.
""")

