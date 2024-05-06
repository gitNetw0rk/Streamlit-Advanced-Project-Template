import streamlit as st

st.set_page_config(
    page_title="Home | Streamlit Advanced Project Template",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={"About": "For updates on template and tutorial follow my LinkedIn> www.linkedin.com/in/marianstancik/"},
)

from sapt import cms, ctrl


def main():
    st.link_button(
        ":red[**Link to GitHub Repository**]",
        "https://github.com/gitNetw0rk/Streamlit-Advanced-Project-Template",
        use_container_width=True,
    )
    # Read and display page content
    cms.render_readme_file_content()

    st.link_button(
        ":red[**Link to GitHub Repository**]",
        "https://github.com/gitNetw0rk/Streamlit-Advanced-Project-Template",
        use_container_width=True,
    )


if __name__ == "__main__":
    ctrl.show_menu()
    main()
