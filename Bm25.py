"""BM25 en Python pur (aucune dépendance) : recherche par mots-clés.

BM25 note chaque chunk selon les mots de la requête qu'il contient :
- un mot rare (ex. un sigle, un numéro d'article) pèse lourd (IDF élevé) ;
- un mot très fréquent (« de », « la ») pèse peu ;
- la longueur du chunk est prise en compte (un mot dans un petit chunk compte plus).
"""
import math
import re
import unicodedata
from collections import Counter


def normalize(text: str) -> str:
    """Minuscules + suppression des accents (« Données » -> « donnees »)."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


# Un mot = lettres/chiffres, avec tirets ou points internes gardés (« 1234-5 », « e-mail »)
_TOKEN = re.compile(r"\w+(?:[-.]\w+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]  # fréquence de chaque mot par chunk
        self.lengths = [len(d) for d in docs]
        self.n = len(docs)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        df = Counter(t for tf in self.tf for t in tf)  # dans combien de chunks apparaît chaque mot
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * self.n
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
            for t in set(query_tokens):
                f = tf.get(t)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return out