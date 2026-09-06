from typing import List, Dict, Any
import uuid

class RAGRetriever:
    def __init__(self,vector_store,embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def add_documents(self,chunks):
        if not chunks:
            raise ValueError("No chunks received")
        texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
        embeddings = (self.embedding_manager.generate_embedding(texts))
        ids = []
        documents = []
        metadatas = []
        for chunk, embedding in zip(chunks,embeddings):
            ids.append(str(uuid.uuid4()))
            documents.append(chunk.page_content)
            metadata = dict(chunk.metadata)
            metadatas.append(metadata)
        self.vector_store.collection.add(ids=ids,documents=documents,embeddings=embeddings.tolist(),metadatas=metadatas)
        print(f"Added {len(documents)} chunks")

    def retrieve(self,query: str,top_k: int = 5,score_threshold: float = 0.3) -> List[Dict[str,Any]]:
        print("Query:",query)
        query_embedding = (
            self.embedding_manager
            .generate_embedding(
                [query]
            )[0]
        )
        results = (
            self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k

            )
        )
        print("CHROMA DOCUMENTS")
        print(results["documents"] )
        print("DISTANCES")
        print(results["distances"])
        if not results["documents"][0]:
            return []
        retrieved_docs = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1 - distance
            if similarity >= score_threshold:
                retrieved_docs.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": float(distance),
                    "cosine_similarity":float(similarity)
                })
        print("Retrieved:",len(retrieved_docs))
        return retrieved_docs