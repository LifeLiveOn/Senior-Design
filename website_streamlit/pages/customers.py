import streamlit as st
import requests
from utils import authorized_request

st.title("Customers")

# ---------------------- STREAMLIT LOGIN CHECK ----------------------
if not st.user.is_logged_in:
    st.error("You must login with Google first.")
    st.stop()

# Make sure JWT exists (it should have been set on login page)
if "access" not in st.session_state:
    st.warning("Your session is not linked to backend yet. Please log in again.")
    st.stop()


# ---------------------- FETCH CUSTOMERS ----------------------
st.header("Your Customers")

res = authorized_request("GET", "/api/customers/")
if res.status_code == 200:
    customers = res.json()
    if customers:
        for c in customers:
            st.write(f"**{c['name']}** — {c['phone']} | {c['address']}")
    else:
        st.info("No customers yet.")
else:
    st.error("Failed to load customers.")


# ---------------------- CREATE CUSTOMER FORM ----------------------
st.header("Create New Customer")

with st.form("create_customer"):
    name = st.text_input("Name")
    phone = st.text_input("Phone")
    address = st.text_area("Address")
    submitted = st.form_submit_button("Create")

if submitted:
    payload = {"name": name, "phone": phone, "address": address}

    res = authorized_request("POST", "/api/customers/", json=payload)

    if res.status_code == 201:
        st.success(f"Customer '{name}' created successfully!")
        st.rerun()
    else:
        st.error(res.json())
