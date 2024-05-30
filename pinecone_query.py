import os
from dotenv import load_dotenv
import logging
from pinecone import Pinecone
from utils.vectorizer import Vectorizer
from langchain_openai import ChatOpenAI

# Disable parallelism in tokenizers to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RAGProcessor:
    def __init__(self):
        load_dotenv()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_index_name = os.getenv('PINECONE_INDEX_NAME').lower().replace(' ', '-')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = os.getenv('MODEL_NAME', 'gpt-4.0-turbo')

        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.pinecone_index_name)

        self.vectorizer = Vectorizer()
        self.chat_model = ChatOpenAI(api_key=self.openai_api_key, model=self.model_name)

    def run_query(self, query):
        embed = self.vectorizer.create_embeddings([query])[0]
        vector = embed.tolist() if hasattr(embed, 'tolist') else embed
        res = self.index.query(vector=vector, top_k=3, include_metadata=True)
        contexts = [f"Document {x['metadata']['filename']} might be relevant." for x in res['matches']]

        prompt = ("You are a helpful assistant. Based on the context provided below, "
                  "generate detailed steps. "
                  "Use the context to guide the user clearly and concisely:\n\n" +
                  "\n\n".join(contexts) +
                  f"\n\nQuestion: {query}\n\nAnswer:")

        response = self.chat_model.invoke(input=prompt, max_tokens=1500)


        if hasattr(response, 'content'):
            return response.content
        return "No detailed instructions generated."

if __name__ == "__main__":
    rag_processor = RAGProcessor()
    query = input("Enter your query: ")
    response = rag_processor.run_query(query)
    print('-' * 80)
    print(response)
