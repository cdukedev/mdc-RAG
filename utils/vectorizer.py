from sentence_transformers import SentenceTransformer

class Vectorizer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_embeddings(self, texts):
        print(f'Creating the embeddings..\n\n')
        embeddings_list = []
        for text in texts:
            embedding = self.model.encode(text, convert_to_tensor=False)
            embeddings_list.append(embedding)
        return embeddings_list
