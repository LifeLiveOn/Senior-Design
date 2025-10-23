import streamlit as st
import pandas as pd 
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Dashboard",
    layout="wide",
    initial_sidebar_state="expanded")

# Get CSV data
df_reshaped = pd.read_csv('../clients.csv', skipinitialspace=True)

# Sidebar
with st.sidebar:
    st.title('Dashboard')
    
    client_list = list(df_reshaped.client_name)
    selected_year = st.selectbox('Select client', client_list)

st.markdown('#### Images')
st.dataframe(df_reshaped,
                 column_order=("client_name", "report_status"),
                 hide_index=True,
                 column_config={
                    "client_name": st.column_config.TextColumn(
                        "Name",
                    ),
                    "report_status": st.column_config.TextColumn(
                        "Status",
                    )}
                 )

# # Session States
# if 'generating_report' not in st.session_state:
#     st.session_state['generating_report'] = False

# # Functions
# def GenerateReport(imageUploaded):
#     st.write(st.session_state.generating_report)
#     if imageUploaded is not None:
#         st.session_state.generating_report = True
#         for image in imageUploaded:
#             print("hello")

# # Main
# uploadedImages = st.file_uploader("Upload Roof Image", type=["jpg", "jpeg"])

# st.button("Generate", disabled=st.session_state.generating_report, on_click=GenerateReport(uploadedImages))
