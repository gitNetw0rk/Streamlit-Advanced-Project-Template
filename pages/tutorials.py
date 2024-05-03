import streamlit as st

st.set_page_config(
    page_title="Tutorials | Streamlit Advanced Project Template",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "About": "For updates on template and tutorial follow my LinkedIn> www.linkedin.com/in/marianstancik/"
    },
)

from sapt import cms, ctrl


def main():
    st.warning("Tutorial content is published in chronologically reversed order.")

    # Read and display page content
    page_content = cms.read_docx_file(file_path="tutorials/sapt_tutorials.docx")
    cms.render_page_content(page_content)


if __name__ == "__main__":
    ctrl.show_menu()
    main()
