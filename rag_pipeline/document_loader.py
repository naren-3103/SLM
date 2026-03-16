import os
from pypdf import PdfReader

def load_documents(folder):

    docs = []

    for file in os.listdir(folder):

        if file.endswith(".pdf"):

            reader = PdfReader(os.path.join(folder, file))

            text = ""

            for page in reader.pages:
                text += page.extract_text()

            docs.append(text)

    return docs