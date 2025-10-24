import streamlit as st
import boto3
import bcrypt
import requests

aws_info = st.secrets["dynamodb"]
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=aws_info["aws_access_key_id"],
    aws_secret_access_key=aws_info["aws_secret_access_key"],
    region_name=aws_info["region_name"]
)
table = dynamodb.Table(aws_info["table_name"])

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

if st.button("Login"):
    if not username or not password:
        st.error("Both fields are required.")
    else:
        try:
            response = table.get_item(Key={'username': username})
            if 'Item' not in response:
                st.error("Username does not exist.")
            else:
                stored_hashed_password = response['Item']['password'].encode('utf-8')
                if bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password):
                    st.success("Login successful!")
                else:
                    st.error("Incorrect password.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("""
    <style>
    .signup-link {
        color: #d11b13;
        font-weight: bold;
        text-decoration: none;
        cursor: pointer;
        transition: 0.3s;
    }
    .signup-link:hover {
        color: #a11b13;
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    '<p>Don\'t have an account? '
    '<a class="signup-link" href="/signup" target="_self">Sign up</a>'
    '</p>', unsafe_allow_html=True
)