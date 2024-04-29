# module CMS (content management system) collect components for delivering content over Streamlit app using standardized document formats (i.e. .docx)
import streamlit as st

from docx import Document


@st.cache_resource(ttl="5d")
def render_readme_file_content():
    # inspiration> https://github.com/avrabyt/Streamlit-Readme-to-App
    with open("README.md", "r") as f:
        readme_lines = f.readlines()

    for line in readme_lines:
        st.markdown(line)


@st.cache_resource(ttl="5d")
def read_docx_file(file_path):
    # Load the .docx file
    doc = Document(file_path)

    # Initialize an empty list to store the content
    content = []

    # Define a mapping of heading styles in docx to HTML heading tags
    HEADING_MAPPING = {
        "Heading 1": "h1",
        "Heading 2": "h2",
        "Heading 3": "h3",
        "Heading 4": "h4",
        "Heading 5": "h5",
        # Add more mappings as needed
    }

    # Iterate through paragraphs in the document and extract text and styling
    for para in doc.paragraphs:
        # Extract text
        para_text = para.text.strip()

        # Extract styling
        para_style = None
        if para.style.name in HEADING_MAPPING:
            para_style = HEADING_MAPPING[para.style.name]
        elif para.runs:
            # Check if the paragraph contains any runs (formatting)
            for run in para.runs:
                # Check if the run has bold, italic, etc.
                if run.bold:
                    para_style = "bold"
                elif run.italic:
                    para_style = "italic"
                # Add more conditions for other formatting styles as needed

        # Append text and styling to the content list
        content.append((para_text, para_style))

    return content


@st.cache_resource(ttl="5d")
def render_page_content(page_content):
    for para_text, para_style in page_content:
        if para_style:
            # Apply styling if present
            st.markdown(f"<{para_style}>{para_text}</{para_style}>", unsafe_allow_html=True)
        else:
            st.write(para_text)
