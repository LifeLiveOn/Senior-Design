import streamlit as st
import requests
import boto3
import bcrypt

aws_info = st.secrets["dynamodb"]

dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=aws_info["aws_access_key_id"],
    aws_secret_access_key=aws_info["aws_secret_access_key"],
    region_name=aws_info["region_name"]
)
table = dynamodb.Table(aws_info["table_name"])

def validate_password(password):
    import re
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*(),.?\":{}|<>]).{8,12}$"
    return bool(re.match(pattern, password))

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

st.title("Sign Up")

username = st.text_input("Username")
password = st.text_input("Password", type="password")
confirm_password = st.text_input("Confirm Password", type="password")

if st.button("Sign Up"):
    if not username or not password or not confirm_password:
        st.error("All fields are required.")
    elif password != confirm_password:
        st.error("Passwords do not match.")
    elif not validate_password(password):
        st.error("Password must be 8-12 characters long, include at least one uppercase letter, one lowercase letter, one digit, and one special character.")
    else:
        try:
            response = table.get_item(Key={'username': username})
            if 'Item' in response:
                st.error("Username already exists. Please choose a different username.")
            else:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                table.put_item(Item={
                    'username': username,
                    'password': hashed_password.decode('utf-8')
                })
                st.success("Account created successfully! You can now log in.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
