import streamlit as st
import pandas as pd 
import numpy as np
import glob
import json
import math
from PIL import Image

CLIENT_COLUMN_COUNT = 1
IMAGE_COLUMN_COUNT = 5
CLIENTS_PER_PAGE = 10

with open("website_streamlit/css/style_dashboard.css") as source:
    design = source.read()

if 'page_number' not in st.session_state:
    st.session_state['page_number'] = 1

def page_next(client_count):
    if st.session_state['page_number'] >= math.ceil(client_count / 10):
        st.session_state['page_number'] = 1
    else:
        st.session_state['page_number'] += 1
    st.rerun()

def page_back(client_count):
    if st.session_state['page_number'] <= 1:
        st.session_state['page_number'] = math.ceil(client_count / 10)
    else:
        st.session_state['page_number'] -= 1
    st.rerun()

@st.dialog('Report', width='large')
def open_report_window(client):
    # Images
    files = glob.glob('website_streamlit/database/images/client' + str(client['index']) + '/*')
    images = []

    for file in files:
        images.append(Image.open(file))

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

# Get client json
with open('website_streamlit/database/client_info.json', 'r') as file:
    client_info = json.load(file)

df = pd.DataFrame(client_info['clients'])
df['date'] = pd.to_datetime(df['date'])

# Page configuration
st.set_page_config(
    page_title='Dashboard',
    layout='wide',
    initial_sidebar_state='expanded')

# Layout
_, column_main, _ = st.columns([1,4,1])

with column_main:
    st.markdown('# Client')

    # Search
    column_search_name, column_search_address = st.columns(2)

    with column_search_name:
        search_name = st.text_input('Search for client name:', placeholder='Search for client name', label_visibility='collapsed')

    with column_search_address:
        search_address = st.text_input('Search for client address:', placeholder='Search for client address', label_visibility='collapsed')
    
    sort = st.radio('Sort by:', ['Date', 'Rank', 'Name', 'Address'], horizontal=True, label_visibility='collapsed')

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

max_index = st.session_state['page_number'] * CLIENTS_PER_PAGE
min_index = max_index - CLIENTS_PER_PAGE
df_within_page = df_search.iloc[min_index:max_index]

# 
client_count = len(df_search)
page_count = math.ceil(client_count / 10)

if (st.session_state['page_number'] > page_count and (search_name or search_address)):
    st.session_state['page_number'] = 1

# Clients
_, column_cards, _ = st.columns([1,6,1])

with column_cards:
# with st.container(height=400,):
    st.markdown('---')

    # Page Controls 1
    _, column_footer, _ = st.columns(3)
    with column_footer:
        columns_page_controls = st.columns([1.5, 1, 1.5])
        
        with columns_page_controls[0]:
            if st.button('◄', width='stretch', key='left0'):
                page_back(client_count)
        with columns_page_controls[2]:
            if st.button('►', width='stretch', key='right0'):
                page_next(client_count)

        with columns_page_controls[1]:
            st.html(f'''
                <style>{design}</style>
                <div class="text_box">
                    <p>{st.session_state['page_number']} of {page_count}</p>
                </div>
            ''')

    # Client Cards
    client_columns = st.columns(CLIENT_COLUMN_COUNT)

    for i, client in df_within_page.iterrows():
        with client_columns[i % CLIENT_COLUMN_COUNT]:
            with st.container(border=True):
                columns_card_info = st.columns([1, 1, 1, 1, 0.5])

                rank = client['rank']
                if rank == -1:
                    rank = '...'

                with columns_card_info[0]:
                    st.html(f'''
                        <style>{design}</style>
                        <h3>{client['date'].date()}</h3>
                    ''')

                with columns_card_info[1]:
                    st.html(f'''
                        <style>{design}</style>
                        <h3>{client['name']}</h3>
                    ''')

                with columns_card_info[2]:
                    st.html(f'''
                        <style>{design}</style>
                        <h3>{client['address']}</h3>
                    ''')

                with columns_card_info[3]:
                    st.html(f'''
                        <style>{design}</style>
                        <h3>{client['number']}</h3>
                    ''')

                with columns_card_info[4]:
                    st.html(f'''
                        <style>{design}</style>
                        <div class="client_card_header">
                            <div class="rank">
                                <h2>{rank}</h2>
                            </div>
                        </div>
                    ''')

                
                if st.button('**View Report**', key=client['index'], type='primary'):
                    open_report_window(client=client)

    # Page Controls 2
    _, column_footer, _ = st.columns(3)
    with column_footer:
        columns_page_controls = st.columns([1.5, 1, 1.5])
        
        with columns_page_controls[0]:
            if st.button('◄', width='stretch', key='left1'):
                page_back(client_count)
        with columns_page_controls[2]:
            if st.button('►', width='stretch', key='right1'):
                page_next(client_count)

        with columns_page_controls[1]:
            st.html(f'''
                <style>{design}</style>
                <div class="text_box">
                    <p>{st.session_state['page_number']} of {page_count}</p>
                </div>
            ''')
