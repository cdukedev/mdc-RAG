import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def preprocess_text(text):
    # Replace consecutive spaces, newlines, and tabs
    text = re.sub(r'\s+', ' ', text)
    return text

class PDFProcessor:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def process_pdfs(self):
        all_texts = []
        all_metadata = []
        print(f'Looking for pdfs...\n\n')

        for filename in os.listdir(self.directory_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.directory_path, filename)
                print(f'Found pdf : {filename}\n\n')
                texts, metadata = self.process_pdf(file_path)
                all_texts.extend(texts)
                all_metadata.extend(metadata)
        return all_texts, all_metadata

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        print(f'Splitting the text into the chunks...\n\n')

        texts = []
        metadata = []
        for page_num, page in enumerate(pages):
            preprocessed_text = preprocess_text(page.page_content)
            chunks = text_splitter.split_text(preprocessed_text)
            for chunk_num, chunk in enumerate(chunks):
                texts.append(chunk)
                metadata.append({
                    'filename': os.path.basename(file_path),
                    'page_number': page_num + 1,
                    'chunk_number': chunk_num + 1,
                    'text': chunk
                })
        return texts, metadata
