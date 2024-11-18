import streamlit as st
from streamlit_option_menu import option_menu
import preprocess_ignb
import app

st.set_page_config(
    page_title='Sentiment Analysis', layout='wide')

with st.sidebar:
    selected = option_menu(
        'Proses',
        ['Analisis Sentimen', 'Klasifikasi Teks'],
        icons=['file-earmark-text', 'file-earmark-text'],
        menu_icon='cast',
        default_index=0,
    )

if selected == 'Analisis Sentimen':
    preprocess_ignb.app()
elif selected == 'Klasifikasi Teks':
    app.main()
