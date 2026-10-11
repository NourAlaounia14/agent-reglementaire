"""Lit les fichiers .txt / .md du dossier docs/, les découpe, et les stocke dans Qdrant.
À lancer :  python ingest.py
"""
from pathlib import Path

from qdrant_client import models

from rag import COLLECTION, client, embed


def chunk(text: str, size: int = 800) -> list[str]:
    """Découpe par paragraphes, en regroupant jusqu'à ~size caractères."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# 1. Lire et découper tous les fichiers
texts, sources = [], []
for f in sorted(Path("docs").iterdir()):
    if f.suffix.lower() not in {".txt", ".md"}:
        continue
    for c in chunk(f.read_text(encoding="utf-8")):
        texts.append(c)
        sources.append(f.name)

if not texts:
    raise SystemExit("Aucun fichier .txt/.md trouvé dans docs/")

# 2. Calculer les embeddings (appel API, par lots)
print(f"Calcul des embeddings pour {len(texts)} extraits...")
vectors = embed(texts)
dim = len(vectors[0])  # dimension détectée automatiquement selon le modèle
print(f"Dimension des vecteurs : {dim}")

# 3. Recréer la collection (obligatoire si on change de modèle : la dimension change)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    COLLECTION,
    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
)

# 4. Stocker vecteur + texte + nom du fichier (par lots)
points = [
    models.PointStruct(id=i, vector=v, payload={"text": t, "source": s})
    for i, (v, t, s) in enumerate(zip(vectors, texts, sources))
]
BATCH = 64
for start in range(0, len(points), BATCH):
    client.upsert(COLLECTION, points=points[start : start + BATCH])
print(f"{len(texts)} extraits indexés depuis docs/")