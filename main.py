import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field

from rag import retrieve

load_dotenv()

# Toute API compatible OpenAI fonctionne (Grok, Gemini, Groq, Mistral, OpenRouter...)
# Pour changer de fournisseur : on modifie uniquement le fichier .env
client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    max_retries=3,  # réessaie automatiquement en cas de 429 (avec attente)
)
MODEL = os.environ["LLM_MODEL"]

app = FastAPI(title="Agent réglementaire - v1 (RAG)")


# ---------- Entrée : le profil de la startup ----------
class StartupProfile(BaseModel):
    nom: str
    secteur: str = Field(examples=["HealthTech", "FinTech", "EdTech"])
    statut_juridique: str = Field(examples=["SAS", "SARL", "micro-entreprise"])
    stade: str = Field(examples=["idée", "MVP", "premiers clients", "levée de fonds"])
    nb_employes: int = 0
    traite_donnees_personnelles: bool = False
    description: str


# ---------- Sortie : le rapport structuré ----------
class Reglementation(BaseModel):
    nom: str
    pourquoi: str
    priorite: str  # haute / moyenne / basse
    action: str
    source: str  # nom du fichier utilisé, ou "connaissance générale (à vérifier)"


class Rapport(BaseModel):
    resume: str
    reglementations: list[Reglementation]


SYSTEM_PROMPT = """Tu es un assistant juridique spécialisé dans la réglementation
des startups en France. À partir du profil fourni, liste les réglementations
applicables (RGPD, droit des sociétés, droit du travail, réglementation sectorielle,
propriété intellectuelle, fiscalité...).

Règles :
- Appuie-toi EN PRIORITÉ sur les extraits fournis dans EXTRAITS. Pour chaque
  réglementation issue d'un extrait, mets dans "source" le nom du fichier.
- Si tu ajoutes une réglementation absente des extraits, mets
  "connaissance générale (à vérifier)" dans "source".
- Ne cite que des textes dont tu es sûr. Si tu hésites, dis-le dans "pourquoi".

Réponds UNIQUEMENT en JSON avec ce format :
{"resume": "...", "reglementations": [{"nom": "...", "pourquoi": "...", "priorite": "haute|moyenne|basse", "action": "...", "source": "..."}]}"""


def build_query(p: StartupProfile) -> str:
    """Transforme le profil en texte de recherche pour la base documentaire."""
    q = f"{p.secteur} {p.statut_juridique} {p.stade} {p.description}"
    if p.traite_donnees_personnelles:
        q += " données personnelles RGPD"
    if p.nb_employes > 0:
        q += " salariés droit du travail"
    return q


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/analyze", response_model=Rapport)
def analyze(profile: StartupProfile):
    try:
        extraits = retrieve(build_query(profile), k=5)
        contexte = "\n\n".join(f"[{e['source']}]\n{e['text']}" for e in extraits) or "(aucun extrait)"

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"PROFIL:\n{profile.model_dump_json()}\n\nEXTRAITS:\n{contexte}",
                },
            ],
            response_format={"type": "json_object"},
        )
        return Rapport.model_validate_json(response.choices[0].message.content)
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Limite du fournisseur LLM atteinte, réessaie dans un instant.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur agent : {e}")