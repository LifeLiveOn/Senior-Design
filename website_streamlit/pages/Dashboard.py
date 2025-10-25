import streamlit as st
import pandas as pd 
import numpy as np
import glob
from PIL import Image

CLIENT_COLUMN_COUNT = 2

with open("website_streamlit/css/style_dashboard.css") as source:
    design = source.read()

def create_client_card_html(name, address, date, index):
    html_content = f'''
        <style>{design}</style>
        <div class="client_card">
            <h2>{name}</h2>
            <p>Address: {address}</p>
            <p>Date: {date}</p>
        </div>
    '''
    return html_content

# Get CSV data
df = pd.read_csv('website_streamlit/database/clients.csv', skipinitialspace=True)

# Page configuration
st.set_page_config(
    page_title='Dashboard',
    layout='wide',
    initial_sidebar_state='expanded')

column_search, column_content = st.columns([0.3, 0.7])

# Search column
with column_search:
    search = st.text_input("Search for a client")
    columns = st.columns(CLIENT_COLUMN_COUNT)

    df_search = df
    if search:
        df_search = df_search[df_search['name'].str.lower().str.startswith(search)]

    for i, client in df_search.iterrows():
        html_content = create_client_card_html(client['name'], client['address'], client['date'], client['index'])
        columns[i % CLIENT_COLUMN_COUNT].html(html_content)
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
