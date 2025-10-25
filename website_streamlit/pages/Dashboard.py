import streamlit as st
import pandas as pd 
import numpy as np
import glob
from PIL import Image

CLIENT_COLUMN_COUNT = 3

def create_client_card(name, address, date, index):
    html_content = f"""
        <div class="client_card">
            <h2>{name}</h2>
            <p>Address: {address}</p>
            <p>Date: {date}</p>
        </div>
    """
    st.html(html_content, width=200)


# Page configuration
st.set_page_config(
    page_title='Dashboard',
    layout='wide',
    initial_sidebar_state='expanded')

column_search, column_content = st.columns([0.3, 0.7])

with column_search:
    st.text_input("Search for a client")

create_client_card('fdsf', 'fsfdsa', 'fdsf', 0)
# IMAGE_COLUMN_COUNT = 5

# def GetClientColmn(clientName, colmnName):
#     return df.loc[df['name'] == clientName, colmnName].iloc[0]

# # Page configuration
# st.set_page_config(
#     page_title='Dashboard',
#     layout='wide',
#     initial_sidebar_state='expanded')

# # Get CSV data
# df = pd.read_csv('./database/clients.csv', skipinitialspace=True)

# # Sidebar
# with st.sidebar:
#     st.title('Dashboard')
    
#     client_list = list(df.name)
#     selectedClient = st.selectbox('Select client', client_list)

# # Get images
# files = glob.glob('./database/images/client' + str(GetClientColmn(selectedClient, 'index')) + '/*')
# images = []

# for file in files:
#     images.append(Image.open(file))

# # Colum Layout
# mainColumn, sideColumn = st.columns([0.7, 0.3])

# # Images
# with mainColumn:
#     st.markdown('# Images')
#     columns = st.columns(IMAGE_COLUMN_COUNT)
#     for i, img in enumerate(images):
#         columns[i % IMAGE_COLUMN_COUNT].image(img)

# # Client Information
# with sideColumn:
#     st.markdown('#### Image Count')
#     st.markdown(len(images))

#     st.markdown('#### Report Status')
#     st.markdown(str(GetClientColmn(selectedClient, 'report_status')))

#     st.markdown('#### Address')
#     st.markdown(GetClientColmn(selectedClient, 'address'))

#     st.markdown('#### Date')
#     st.markdown(GetClientColmn(selectedClient, 'date'))
