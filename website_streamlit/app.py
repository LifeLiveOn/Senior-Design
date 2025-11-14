import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/api"

st.title("Roof Vision AI")
st.subheader("By Good Neighbor")

res = requests.get(f"{BACKEND_URL}/")

message = res.json()
st.write("Success connection to backend:")
