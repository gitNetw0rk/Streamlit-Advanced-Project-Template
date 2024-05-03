import streamlit as st

st.set_page_config(
    page_title="Demo codes | Streamlit Advanced Project Template",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "About": "For updates on template and tutorial follow my LinkedIn> www.linkedin.com/in/marianstancik/"
    },
)
from sapt import ctrl
import pdis.resume_as_dictionaire as last_tutorial_code


def main():
    last_tutorial_code.main()


if __name__ == "__main__":
    ctrl.show_menu()
    main()
