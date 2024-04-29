# module CTRL (control) collect components controlling app's workflow)
import streamlit as st


def show_menu():
    st.sidebar.page_link(page="home.py", label="Home")
    st.sidebar.divider()
    st.sidebar.page_link(page="pages/demo_pdis.py", label="Last demo")
    st.sidebar.divider()
    st.sidebar.page_link(page="pages/tutorials.py", label="Tutorials")
