import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# Doit être appelé AVANT de lire les variables d'environnement ci-dessous
load_dotenv()

COLLECTION = "reglementation"

# Qdrant Cloud si QDRANT_URL est défini dans .env, sinon stockage local (qdrant_data/)
if os.getenv("QDRANT_URL"):
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=60,
    )
else:
    client = QdrantClient(path="qdrant_data")

# Embeddings via une API compatible OpenAI (aucun modèle local, pas de torch).
# Par défaut on réutilise le fournisseur du LLM ; EMBED_API_KEY / EMBED_BASE_URL
# permettent d'en utiliser un autre uniquement pour les embeddings.
embed_client = OpenAI(
    api_key=os.getenv("EMBED_API_KEY") or os.environ["LLM_API_KEY"],
    base_url=os.getenv("EMBED_BASE_URL") or os.environ["LLM_BASE_URL"],
    max_retries=5,  # réessaie automatiquement en cas de 429
)
EMBED_MODEL = os.environ["EMBED_MODEL"]


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Transforme des textes en vecteurs via l'API, par lots."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = embed_client.embeddings.create(
            model=EMBED_MODEL, input=texts[i : i + batch_size]
        )
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Retourne les k extraits les plus proches de la question."""
    if not client.collection_exists(COLLECTION):
        return []
    res = client.query_points(COLLECTION, query=embed([query])[0], limit=k)
    return [
        {"text": p.payload["text"], "source": p.payload["source"], "score": p.score}
        for p in res.points
    ]