import os
import argparse
from dotenv import load_dotenv
from utils.pdf_processor import PDFProcessor
from utils.vectorizer import Vectorizer
from utils.pinecone_manager import PineconeManager

# Load environment variables from .env file
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='PDF Course Embedding Tool')
    # Add arguments for running the script
    parser.add_argument('--pdf_directory', type=str, default=os.getenv('PDF_DIRECTORY'))
    parser.add_argument('--pinecone_api_key', type=str, default=os.getenv('PINECONE_API_KEY'))
    parser.add_argument('--pinecone_environment', type=str, default=os.getenv('PINECONE_ENVIRONMENT'))
    parser.add_argument('--pinecone_index_name', type=str, default=os.getenv('PINECONE_INDEX_NAME'))
    parser.add_argument('--pinecone_region', type=str, default=os.getenv('PINECONE_REGION'))

    args = parser.parse_args()

    # Initialize components
    pdf_processor = PDFProcessor(directory_path=args.pdf_directory)
    vectorizer = Vectorizer()
    pinecone_manager = PineconeManager(api_key=args.pinecone_api_key, 
                                       cloud=args.pinecone_environment,
                                       region=args.pinecone_region,
                                       index_name=args.pinecone_index_name)

    # Process PDFs and prepare metadata
    texts, metadata = pdf_processor.process_pdfs()  # Corrected method call

    # Create embeddings for all texts
    embeddings = vectorizer.create_embeddings(texts)
    ids = [f"doc_{i}" for i in range(len(embeddings))]  # Assign unique IDs for each document

    # Upsert the embeddings to Pinecone
    pinecone_manager.upsert_embeddings(embeddings, ids, metadata)

if __name__ == "__main__":
    main()
