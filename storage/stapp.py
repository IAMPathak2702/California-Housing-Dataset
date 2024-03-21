import streamlit as st 
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np


st.set_page_config(
    page_title="California Dreaming: Unveiling the Future of Home Prices in the Golden State",
    page_icon="🏠",
    layout="wide"
)

st.markdown('<h1 align=center>Forecasting the Next Wave: Predicting California Houses Prices </h1>',unsafe_allow_html=True)

# Load data
@st.cache_data()
def load_data():
    X_t = pd.read_csv("storage/X_train.csv")
    X_t.drop("Unnamed: 0" , axis=1, inplace=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_t)
    return scaler, X_train

scaler, X_train = load_data()

# Load model
@st.cache_resource()
def load_model():
    model = joblib.load("storage/regmodel.pkl")
    return model

model = load_model()

col1, col2 = st.columns(2)

col1.image("storage/Screenshot 2024-03-21 211958.png")

with col2.form("Please enter parameters to Predict a price"):
    med_inc = st.number_input("Enter Median Income", min_value=0.0, step=1000.0,max_value=10000000.0)

    house_age = st.number_input("Enter House Age", min_value=0, step=1, max_value=60)

    ave_rooms = st.number_input("Enter Average Rooms", min_value=0, step=1, max_value=150)

    ave_bedrooms = st.number_input("Enter Average Bedrooms", min_value=0.0, step=1.0, max_value=50.0)

    population = st.number_input("Enter Population", min_value=0, step=1000, max_value=36000)

    ave_occup = st.number_input("Enter Average Occupation", min_value=0.0, step=1.0, max_value=1300.0)

    latitude = st.number_input("Enter Latitude",placeholder="32.54 to 41.95", min_value=32.54, max_value=41.95)

    longitude = st.number_input("Enter Longitude",placeholder="-120.0 to -114.0", min_value=-120.0, max_value=-114.0)

    predict= st.form_submit_button("PREDICT")
    reset = st.form_submit_button("Reset")
    
# Reset form fields if reset button is clicked
if "Reset" in st.session_state:
    st.experimental_rerun()

if predict:
    input_data = scaler.transform([[med_inc/10000, house_age, ave_rooms, ave_bedrooms, population, ave_occup, latitude, longitude]])
    price = model.predict(input_data)
    with st.spinner():
        html_code = f"<h1 style='font-size: 36pt; color: red;'>Predicted price is: ${np.round(price[0]*10000,2)}</h1>"
        st.markdown(html_code, unsafe_allow_html=True)
