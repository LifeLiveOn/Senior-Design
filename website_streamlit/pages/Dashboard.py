import streamlit as st
import pandas as pd 
import numpy as np
import glob
from PIL import Image

IMAGE_COLUMN_COUNT = 5

def GetClientColmn(clientName, colmnName):
    return df.loc[df['name'] == clientName, colmnName].iloc[0]

# Page configuration
st.set_page_config(
    page_title='Dashboard',
    layout='wide',
    initial_sidebar_state='expanded')

# Get CSV data
df = pd.read_csv('./database/clients.csv', skipinitialspace=True)

# Sidebar
with st.sidebar:
    st.title('Dashboard')
    
    client_list = list(df.name)
    selectedClient = st.selectbox('Select client', client_list)

# Get images
files = glob.glob('./database/images/client' + str(GetClientColmn(selectedClient, 'index')) + '/*')
images = []

for file in files:
    images.append(Image.open(file))

# Colum Layout
mainColumn, sideColumn = st.columns([0.7, 0.3])

# Images
with mainColumn:
    st.markdown('# Images')
    columns = st.columns(IMAGE_COLUMN_COUNT)
    for i, img in enumerate(images):
        columns[i % IMAGE_COLUMN_COUNT].image(img)

# Client Information
with sideColumn:
    st.markdown('#### Image Count')
    st.markdown(len(images))

    st.markdown('#### Report Status')
    st.markdown(str(GetClientColmn(selectedClient, 'report_status')))

    st.markdown('#### Address')
    st.markdown(GetClientColmn(selectedClient, 'address'))

    st.markdown('#### Date')
    st.markdown(GetClientColmn(selectedClient, 'date'))

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
