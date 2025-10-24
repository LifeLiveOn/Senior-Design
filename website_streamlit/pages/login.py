# import streamlit as st
# import requests

# st.title("Login")
# email = st.text_input("Email")
# password = st.text_input("Password", type="password")



# st.markdown("""
#  <​style>
#  div.stButton > button {
#  background-color: #FF6B6B;
#  color: white;
#  border-radius: 8px;
#  border: none;
#  padding: 0.6em 1em;
#  font-weight: bold;
#  transition: 0.3s;
#  }
#  div.stButton > button:hover {
#  background-color:#FF3B3B;
#  transform: scale(1.05);
#  }
#  <​/style>
# """, unsafe_allow_html=True)

# # Test buttons


# st.button("Login")
#     #response = requests.post("http://backend:8000/login", json={"email": email, "password": password})
# st.markdown("Don't have an account? Sign up")




import streamlit as st
import requests

st.title("Login")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

st.markdown("""
<style>
div.stButton > button {
    background-color: #d11b13;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1em;
    font-weight: bold;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #a11b13;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

st.button("Login")
st.markdown("Don't have an account? Sign up")
