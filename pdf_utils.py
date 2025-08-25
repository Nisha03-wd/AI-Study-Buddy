from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    if uploaded_file is None:
        return ""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text