from typing import List
import os
import uuid
import chromadb


class vector_store:
    def __init__(self,collection_name="pdf_documents",persistent_directory="./data/vector_db"):
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.client = None
        self.collection = None
        self.initialize_store()

    def initialize_store(self):
        os.makedirs(self.persistent_directory,exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persistent_directory)
        self.collection = (self.client.get_or_create_collection(name=self.collection_name,metadata={"description":"PDF embeddings for RAG"}))
        print("Vector Store Initialized")
        print("Documents:",self.collection.count())

    def add_documents(self,documents,embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings count mismatch")
        ids = []
        texts = []
        metadatas = []
        vectors = []
        for doc, embedding in zip(documents,embeddings):
            ids.append( str(uuid.uuid4()))
            texts.append(doc.page_content)
            metadata = dict(doc.metadata)
            metadatas.append(metadata)
            vectors.append(embedding.tolist())
        self.collection.add(ids=ids,documents=texts,embeddings=vectors,metadatas=metadatas)
        print(f"Added {len(ids)} chunks")

    def clear(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            print("Old collection deleted")
        except Exception:
            pass

        self.collection = (self.client.create_collection(name=self.collection_name,metadata={"description":"PDF embeddings for RAG"}))
        print("Vector store cleared")

    def count(self):
        return self.collection.count()