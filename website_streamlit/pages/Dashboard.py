import streamlit as st
import pandas as pd 
import numpy as np
import glob
from PIL import Image

CLIENT_COLUMN_COUNT = 2

with open("website_streamlit/css/style_dashboard.css") as source:
    design = source.read()

def create_client_card_html(name, address, date, status, rank, index):
    if rank == -1:
        rank = 'N/A'

    if status == 1:
        status = 'Generating'
        color = '#FFA500'
    elif status == 2:
        status = 'Ready'
        color = "#00EE00"
    else:
        status = 'Needs Images'
        color = '#EE0000'
        
    html_content = f'''
        <style>{design}</style>
        <div class="client_card">
            <div>
                <h2>{name}</h2>
                <div class="rank">
                    <h2>{rank}</h2>
                </div>
            </div>
            <p>{address}</p>
            <p>{date}</p>
            <h4 style="color: {color}">{status}</h4>
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

column_search, column_content = st.columns([0.3, 0.7], border=True)

# Search column
with column_search:
    st.markdown('# Client')

    # Search
    column_input, column_sort = st.columns([0.75, 0.25])

    with column_input:
        search_name = st.text_input("Search for a client name:")

    search_address = st.text_input("Search for a client address:")

    with column_sort:
        sort = st.selectbox('Sort by:', ['Date', 'Name', 'Rank'])

    # Change dataframe
    ascending = True
    if sort.lower() == 'rank':
        ascending = False

    df_sorted = df.sort_values(by=sort.lower(), ascending=ascending)
    df_search = df_sorted

    if search_name:
        df_search = df_search[df_search['name'].str.lower().str.contains(search_name.lower())]

    if search_address:
        df_search = df_search[df_search['address'].str.lower().str.contains(search_address.lower())]

    # Clients
    with st.container(height=400):
        client_columns = st.columns(CLIENT_COLUMN_COUNT)

        for i, client in df_search.iterrows():
            html_content = create_client_card_html(name=client['name'], address=client['address'], date=client['date'], status=client['status'], rank=client['rank'], index=client['index'])
            with client_columns[i % CLIENT_COLUMN_COUNT]:
                st.html(html_content)

# Content Column
with column_content:
    st.markdown('# Report')
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
