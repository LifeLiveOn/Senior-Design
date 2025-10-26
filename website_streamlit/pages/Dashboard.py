import streamlit as st
import pandas as pd 
import numpy as np
import glob
from PIL import Image

CLIENT_COLUMN_COUNT = 2
IMAGE_COLUMN_COUNT = 5

with open("website_streamlit/css/style_dashboard.css") as source:
    design = source.read()

def create_client_card_html(client):
    rank = client['rank']
    status = client['status']

    if rank == -1:
        rank = '...'

    if status == 1:
        status = 'Generating'
        color = '#FFA500'
    elif status == 2:
        status = 'Ready'
        color = "#00EE00"
    else:
        status = 'Needs Images'
        color = '#EE0000'
    
    card_html = f'''
        <style>{design}</style>
        <div class="client_card_header">
            <h3>{client['date']}</h3>
            <div class="rank">
                <h2>{rank}</h2>
            </div>
        </div>
        <p>{client['name']}</p>
        <p>{client['address']}</p>
        <h4 style="color: {color}">{status}</h4>
    '''

    return card_html

def update_report(report_column, client):
    # Images
    files = glob.glob('website_streamlit/database/images/client' + str(client['index']) + '/*')
    images = []

    for file in files:
        images.append(Image.open(file))

    with report_column:
        # Summary
        with st.container(border=True):
            st.markdown('## Summary')

            st.write(f'**Name:** {client['name']}')
            st.write(f'**Address:** {client['address']}')

        # Images
        with st.container(border=True):
            st.markdown('## Images')

            with st.expander(str(len(images))):
                with st.container(height=380, border=False):
                    columns = st.columns(IMAGE_COLUMN_COUNT)

                    for i, img in enumerate(images):
                        columns[i % IMAGE_COLUMN_COUNT].image(img)


# Get CSV data
df = pd.read_csv('website_streamlit/database/clients.csv', skipinitialspace=True)

# Page configuration
st.set_page_config(
    page_title='Dashboard',
    layout='wide',
    initial_sidebar_state='expanded')

# Layout
columns_header = st.columns([0.35, 0.65], gap='large')
column_search, column_content = st.columns([0.35, 0.65], gap='large', border=True)

with columns_header[0]:
    st.markdown('# Client')

with columns_header[1]:
    st.markdown('# Report')


# Search column
with column_search:
    # Search
    column_input, column_sort = st.columns([0.75, 0.25])

    with column_input:
        search_name = st.text_input("Search for client name:")
        search_address = st.text_input("Search for client address:")

    with column_sort:
        sort = st.radio('Sort by:', ['Date', 'Rank', 'Status', 'Name', 'Address'], horizontal=True)

    # Change dataframe
    ascending = True
    if sort.lower() == 'date' or sort.lower() == 'rank':
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
            with client_columns[i % CLIENT_COLUMN_COUNT]:
                with st.container(border=True):
                    card_html = create_client_card_html(client=client)
                    st.html(card_html)

                    if st.button('View', key=client['index']):
                        update_report(column_content, client=client)

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