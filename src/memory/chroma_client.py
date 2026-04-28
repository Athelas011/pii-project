import chromadb
from chromadb.api.types import EmbeddingFunction

from src.config.settings import CHROMA_DB_PATH

# Sentinel: tell chromadb we supply embeddings ourselves, skip any stored function.
_NO_OP_EF: EmbeddingFunction = None  # type: ignore[assignment]


class ChromaClient:
    """Read-only interface to the pre-populated Chroma collections."""

    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        # embedding_function=None so chromadb does not try to instantiate
        # the "default" function stored in collection metadata.
        self._text_col = client.get_collection(
            "text_collection", embedding_function=_NO_OP_EF
        )
        self._image_col = client.get_collection(
            "image_collection", embedding_function=_NO_OP_EF
        )

    def query_text_collection(
        self, embedding: list[float], top_k: int
    ) -> list[dict]:
        return self._query(self._text_col, embedding, top_k)

    def query_image_collection(
        self, embedding: list[float], top_k: int
    ) -> list[dict]:
        return self._query(self._image_col, embedding, top_k)

    @staticmethod
    def _query(collection, embedding: list[float], top_k: int) -> list[dict]:
        raw = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]
        return [
            {"id": id_, "document": doc, "metadata": meta, "distance": dist}
            for id_, doc, meta, dist in zip(ids, docs, metas, dists)
        ]
