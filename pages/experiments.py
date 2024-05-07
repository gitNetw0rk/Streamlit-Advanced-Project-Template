import streamlit as st

st.set_page_config(
    page_title="Experimental apps | Streamlit Advanced Project Template",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={"About": "For updates on template and tutorial follow my LinkedIn> www.linkedin.com/in/marianstancik/"},
)
from sapt import ctrl
import experiments.pfa_with_llms as last_experimental_app


def main():
    last_experimental_app.main()


if __name__ == "__main__":
    ctrl.show_menu()
    main()
