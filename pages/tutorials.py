import streamlit as st

from sapt import cms, ctrl


def main():
    st.warning("Tutorial content is published in chronologically reversed order.")

    # Read and display page content
    page_content = cms.read_docx_file(file_path="tutorials/sapt_tutorials.docx")
    cms.render_page_content(page_content)


if __name__ == "__main__":
    ctrl.show_menu()
    main()
