import os
import logging
import traceback
import pinecone
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

class PineconeManager:
    def __init__(self, api_key, cloud='aws', region='us-west-2', index_name='course-vectors'):
        load_dotenv()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.api_key = api_key
        self.cloud = cloud
        self.region = region
        self.index_name = index_name.replace(' ', '-').lower()  # Ensure the index name is in the correct format
        
        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.ensure_index_exists()
        except Exception as e:
            logging.error("Initialization failed: %s\n%s", str(e), traceback.format_exc())
            raise

    def ensure_index_exists(self):
        config = {
            "dimension": 384,  # Adjust dimension according to the model output
            "metric": "cosine"
        }
        spec = ServerlessSpec(cloud=self.cloud, region=self.region)

        try:
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                logging.info("Creating Pinecone index: %s", self.index_name)
                self.pc.create_index(
                    name=self.index_name, 
                    dimension=config['dimension'], 
                    metric=config['metric'], 
                    spec=spec
                )
            else:
                logging.info("Pinecone index already exists: %s", self.index_name)
            self.index = self.pc.Index(self.index_name)  # Ensure the index attribute is set here
        except Exception as e:
            logging.error("Failed to ensure index exists: %s\n%s", str(e), traceback.format_exc())

    def upsert_embeddings(self, embeddings, ids, metadata, batch_size=100):
        total_batches = len(embeddings) // batch_size + (1 if len(embeddings) % batch_size else 0)
        
        for i in tqdm(range(0, len(embeddings), batch_size)):
            start = i
            end = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[start:end]
            batch_metadata = metadata[start:end]
            batch_ids = ids[start:end]
            batch_vectors = [(batch_ids[j], batch_embeddings[j], batch_metadata[j]) for j in range(len(batch_embeddings))]
            
            try:
                self.index.upsert(vectors=batch_vectors)
                logging.info(f"Upserted batch {i // batch_size + 1}/{total_batches} to Pinecone index: {self.index_name}")
            except Exception as e:
                logging.error("Error during upsert operation: %s\n%s", str(e), traceback.format_exc())
                raise
