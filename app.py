import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

data = pd.read_csv("job_prediction .csv")

st.title("CAREER PREDICTION")
st.write("Enter STUDENT DETAILS:")

stu = {}

for col in data.columns:
    if col != "ROLE":
        stu[col] = st.selectbox(col, list(encoders[col].classes_))

if st.button("PREDICT"):

    input_data = pd.DataFrame([stu])

    for col in input_data.columns:
        input_data[col] = encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)

    role = encoders["ROLE"].inverse_transform(prediction)

    st.success("PREDICTED CAREER")
    st.write(role[0])
