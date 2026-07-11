import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

df = pd.read_csv("dataset9000.csv")

st.title("Alumni Career Prediction System")
st.write("Enter the student details below")

student = {}

for col in df.columns:
    if col != "Role":
        student[col] = st.selectbox(col, list(encoders[col].classes_))

if st.button("Predict Role"):

    input_df = pd.DataFrame([student])

    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)

    role = encoders["Role"].inverse_transform(prediction)

    st.success("Predicted Career Role")
    st.write(role[0])
