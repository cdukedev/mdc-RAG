
# MDC-RAG

This project demonstrates the use of Pinecone for creating a vector embedding database and Langchain for handling language operations with OpenAI's ChatModel. The goal is to manage coursework from PDFs, create an index, and query the data efficiently.

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone [<repository_url>](https://github.com/VirajDeshwal/mdc-RAG)
   cd mdc-RAG
   ```

2. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   - Add your OpenAI and Pinecone API keys to .env file:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     ```

4. **Add your PDF files:**
   - Place the PDF files you want to process in the `input_src` folder.

5. **Create the vector database and index the data:**
   ```sh
   python run.py
   ```

6. **Run a query against the indexed data:**
   ```sh
   python pinecone_query.py
   ```

## Project Structure

- `run.py`: Script to create the vector index and store the data.
- `pinecone_query.py`: Script to query the Pinecone vector database.
- `requirements.txt`: List of dependencies required for the project.
- `input_src/`: Directory to place the PDF files to be processed.
- `utils/`: Utility functions and modules used in the project.
- `.env`: Environment file to store API keys (not included in the repository).

## Usage

1. **Indexing Data:**
   - Ensure your PDFs are in the `input_src` folder.
   - Run `python run.py` to create the vector index.

2. **Querying Data:**
   - Run `python pinecone_query.py` to perform queries on the indexed data.

## Notes

- Make sure to replace `your_openai_api_key` and `your_pinecone_api_key` with your actual API keys in the `.env` file.

