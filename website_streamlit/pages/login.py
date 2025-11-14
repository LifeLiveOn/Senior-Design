import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/auth/google/"

st.title("Login with Google")

# If user not logged in via Streamlit
if not st.user.is_logged_in:
    if st.button("Login with Google"):
        st.login("google")
    st.stop()

# Logged in
user_info = st.user
email = user_info.get("email")
name = user_info.get("name", "")
picture = user_info.get("picture", "")

st.write(f"Hello, {email}")

if st.button("Log out"):
    st.logout()
    st.rerun()

# Send user data to backend
with st.spinner("Verifying with backend…"):
    res = requests.post(API_URL, json={"email": email, "name": name})

if res.status_code == 200:
    data = res.json()
    st.session_state["access"] = data["access"]
    st.session_state["refresh"] = data["refresh"]
    st.session_state["email"] = email
    st.session_state["name"] = name
    st.session_state["picture"] = picture
    st.success("Login successful!")
    # st.switch_page("pages/customers.py")
else:
    st.error("Failed to verify login with backend.")
