import streamlit as st
import pandas as pd 
import numpy as np
import glob
import json
import math
from PIL import Image

IMAGE_COLUMN_COUNT = 5
CUSTOMERS_PER_PAGE = 5

with open("website_streamlit/css/style_dashboard.css") as source:
    design = source.read()

if 'page_number' not in st.session_state:
    st.session_state['page_number'] = 1

def page_next(customer_count):
    if st.session_state['page_number'] >= math.ceil(customer_count / 10):
        st.session_state['page_number'] = 1
    else:
        st.session_state['page_number'] += 1
    st.rerun()

def page_back(customer_count):
    if st.session_state['page_number'] <= 1:
        st.session_state['page_number'] = math.ceil(customer_count / 10)
    else:
        st.session_state['page_number'] -= 1
    st.rerun()

@st.dialog('Customer', width='large')
def open_report_window(customer):
    # Images
    files = glob.glob('website_streamlit/database/images/customer' + str(customer['id']) + '/*')
    images = []

    for file in files:
        images.append(Image.open(file))

    if len(images) == 0:
        st.html('<h1 style="text-align: center;">Waiting for images...</h1>')
    else:
        # Tabs
        tab_report, tab_settings = st.tabs(['Report', 'Settings'])

        with tab_report:
            # Summary
            with st.container(border=True):
                st.markdown('## Summary')

                st.write(f'**Name:** {customer['name']}')
                st.write(f'**Address:** {customer['address']}')

            # Images
            with st.container(border=True):
                st.markdown('## Images')

                with st.expander(str(len(images))):
                    with st.container(height=380, border=False):
                        columns = st.columns(IMAGE_COLUMN_COUNT)

                        for i, img in enumerate(images):
                            columns[i % IMAGE_COLUMN_COUNT].image(img)

        with tab_settings:
            columns_settings_row1 = st.columns([1, 3, 2], border=True)

            # Mode
            with columns_settings_row1[0]:
                infer_mode = st.radio("Select inference mode:", ["Normal", "Tiled"])
            
            # Threshold
            with columns_settings_row1[1]:
                conf_threshold = st.slider(
                    "Confidence Threshold:",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.4,
                    step=0.05,
                    help="Minimum confidence required for a detection to be considered valid."
                )

            # Tile size
            with columns_settings_row1[2]:
                tile_size_option = st.selectbox(
                    "Tile size (for tiled mode only):",
                    ["Tiny", "Small", "Normal", "Large"],
                    index=2
                )

            tile_size_map = {
                "Tiny": 224,
                "Small": 448,
                "Normal": 560,
                "Large": 616,
            }
            tile_size = tile_size_map[tile_size_option]

            # Selected Images
            with st.expander('Selected Images'):
                # with st.container(height=380, border=False):
                columns = st.columns(IMAGE_COLUMN_COUNT)

                for i, img in enumerate(images):
                    columns[i % IMAGE_COLUMN_COUNT].image(img)
                    
            # Regenerate
            st.button('Regenerate Report', type='primary')


# Get customer json
with open('website_streamlit/database/customer_info.json', 'r') as file:
    customer_info = json.load(file)

df = pd.DataFrame(customer_info['customers'])
df['date'] = pd.to_datetime(df['date'])

# Page configuration
st.set_page_config(
    page_title='Dashboard',
    layout='wide',
    initial_sidebar_state='expanded')

# Layout
_, column_header, _ = st.columns([1,6,1])
with column_header:
    st.markdown('<h1 style="text-align: center;">Customers</h1>', unsafe_allow_html=True)
    st.markdown('---')

_, column_main, _ = st.columns([1,4,1])

with column_main:
    # Search
    column_search_name, column_search_address = st.columns(2)

    with column_search_name:
        search_name = st.text_input('Search for customer name:', placeholder='Search for customer name', label_visibility='collapsed')

    with column_search_address:
        search_address = st.text_input('Search for customer address:', placeholder='Search for customer address', label_visibility='collapsed')
    
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

max_index = st.session_state['page_number'] * CUSTOMERS_PER_PAGE
min_index = max_index - CUSTOMERS_PER_PAGE
df_within_page = df_search.iloc[min_index:max_index]

# Counts
customer_count = len(df_search)
page_count = math.ceil(customer_count / 10)

if (st.session_state['page_number'] > page_count and (search_name or search_address)):
    st.session_state['page_number'] = 1
    st.rerun()

# Customers
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
                page_back(customer_count)
        with columns_page_controls[2]:
            if st.button('►', width='stretch', key='right0'):
                page_next(customer_count)

        with columns_page_controls[1]:
            st.html(f'''
                <style>{design}</style>
                <div class="text_box">
                    <p>{st.session_state['page_number']} of {page_count}</p>
                </div>
            ''')

    # customer Cards
    for i, customer in df_within_page.iterrows():
        with st.container(border=True):
            columns_card_info = st.columns([1, 1, 1, 1, 0.5])

            rank = customer['rank']
            if rank == -1:
                rank = '...'

            with columns_card_info[0]:
                st.html(f'''
                    <style>{design}</style>
                    <h3>Date:</h3>
                    <p>{customer['date'].date()}</p>
                ''')

            with columns_card_info[1]:
                st.html(f'''
                    <style>{design}</style>
                    <h3>Name:</h3>
                    <p>{customer['name']}</p>
                ''')

            with columns_card_info[2]:
                st.html(f'''
                    <style>{design}</style>
                    <h3>Address:</h3>
                    <p>{customer['address']}</p>
                ''')

            with columns_card_info[3]:
                st.html(f'''
                    <style>{design}</style>
                    <h3>Number:</h3>
                    <p>{customer['number']}</p>
                ''')

            with columns_card_info[4]:
                st.html(f'''
                    <style>{design}</style>
                    <div class="customer_card_header">
                        <div class="rank">
                            <h2>{rank}</h2>
                        </div>
                    </div>
                ''')
                # st.markdown(":blue-badge[Hail Damage] :gray-badge[Wind Damage]")
            
            if st.button('**View Report**', key=customer['id'], type='primary'):
                open_report_window(customer=customer)

    # Page Controls 2
    _, column_footer, _ = st.columns(3)
    with column_footer:
        columns_page_controls = st.columns([1.5, 1, 1.5])
        
        with columns_page_controls[0]:
            if st.button('◄', width='stretch', key='left1'):
                page_back(customer_count)
        with columns_page_controls[2]:
            if st.button('►', width='stretch', key='right1'):
                page_next(customer_count)

        with columns_page_controls[1]:
            st.html(f'''
                <style>{design}</style>
                <div class="text_box">
                    <p>{st.session_state['page_number']} of {page_count}</p>
                </div>
            ''')
