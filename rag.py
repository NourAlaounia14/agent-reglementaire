import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from Bm25 import BM25, tokenize

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


@lru_cache(maxsize=256)
def _embed_query(query: str) -> tuple[float, ...]:
    """Embedding d'une requête, mis en cache (même requête = pas de nouvel appel API)."""
    return tuple(embed([query])[0])


# ---------------------------------------------------------------------------
# 1. Recherche VECTORIELLE (sens) : Qdrant compare les vecteurs (cosinus)
# ---------------------------------------------------------------------------
def _vector_search(query: str, n: int) -> list[dict]:
    res = client.query_points(COLLECTION, query=list(_embed_query(query)), limit=n)
    return [
        {"id": p.id, "text": p.payload["text"], "source": p.payload["source"], "score": p.score}
        for p in res.points
    ]


# ---------------------------------------------------------------------------
# 2. Recherche par MOTS-CLÉS (BM25) : index construit en mémoire depuis Qdrant
# ---------------------------------------------------------------------------
_corpus: dict | None = None


def _load_corpus() -> dict:
    """Lit tous les chunks dans Qdrant (une fois) et construit l'index BM25."""
    global _corpus
    if _corpus is None:
        items, offset = [], None
        while True:
            points, offset = client.scroll(
                COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            items += [
                {"id": p.id, "text": p.payload["text"], "source": p.payload["source"]}
                for p in points
            ]
            if offset is None:
                break
        _corpus = {"items": items, "bm25": BM25([tokenize(i["text"]) for i in items])}
    return _corpus


def _bm25_search(query: str, n: int) -> list[dict]:
    corpus = _load_corpus()
    if not corpus["items"]:
        return []
    scores = corpus["bm25"].scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [{**corpus["items"][i], "score": scores[i]} for i in order if scores[i] > 0]


# ---------------------------------------------------------------------------
# 3. FUSION : Reciprocal Rank Fusion (RRF)
#    On ne compare pas les scores (échelles différentes), seulement les RANGS :
#    un chunk bien classé dans une liste ou dans l'autre gagne des points.
# ---------------------------------------------------------------------------
def _rrf(rankings: list[list[dict]], k_rrf: int = 60) -> list[dict]:
    fused: dict = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            entry = fused.setdefault(item["id"], {**item, "score": 0.0})
            entry["score"] += 1 / (k_rrf + rank)
    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)


def retrieve(query: str, k: int = 5, mode: str = "hybrid", n_candidates: int = 20) -> list[dict]:
    """Retourne les k meilleurs extraits.

    mode = "vector" (sens), "bm25" (mots-clés) ou "hybrid" (les deux, fusionnés par RRF).
    Attention : le champ "score" n'a pas la même signification selon le mode
    (cosinus, score BM25, ou score RRF) : ne compare pas les scores entre modes.
    """
    if not client.collection_exists(COLLECTION):
        return []
    if mode == "vector":
        return _vector_search(query, k)
    if mode == "bm25":
        return _bm25_search(query, k)
    fused = _rrf([_vector_search(query, n_candidates), _bm25_search(query, n_candidates)])
    return fused[:k]