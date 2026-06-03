import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "MTGCards"


class VectorSearcher:
    def __init__(self, data_path: str = CHROMA_DATA_PATH,
                 collection_name: str = COLLECTION_NAME):
        self._client = chromadb.PersistentClient(path=data_path)
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self._collection = self._client.get_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
        )

    def search(self, query_text: str, n_results: int = 10) -> dict:
        return self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )

    def suggest_indices(self, query_text: str, n_results: int = 10) -> list[int]:
        results = self.search(query_text, n_results)
        return [int(id_.replace("id", "")) for id_ in results["ids"][0]]

    def suggest_documents(self, query_text: str, n_results: int = 10) -> list[str]:
        results = self.search(query_text, n_results)
        return results["documents"][0]
