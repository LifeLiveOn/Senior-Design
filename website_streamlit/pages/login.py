import streamlit as st
import requests

st.title("Login")
email = st.text_input("Email")
password = st.text_input("Password", type="password")
st.button("Login")
    #response = requests.post("http://backend:8000/login", json={"email": email, "password": password})

