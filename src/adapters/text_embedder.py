from openai import OpenAI

from src.config.settings import OPENAI_API_KEY, TEXT_EMBEDDING_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str) -> list[float]:
    """Embed text with OpenAI text-embedding-3-small (1536-dim).

    Matches the model used when the text_collection was originally populated.
    """
    response = _client.embeddings.create(model=TEXT_EMBEDDING_MODEL, input=text)
    return response.data[0].embedding
