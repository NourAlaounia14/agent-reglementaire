"""Évaluation du RAG.

    python eval.py                  # qualité de la RECHERCHE (recall@k, MRR) : vector / bm25 / hybrid
    python eval.py --faithfulness   # + FIDÉLITÉ des réponses aux extraits (appelle le LLM, consomme du quota)

Le jeu de test est dans eval_set.json (voir eval_set.example.json).
"""
import json
import sys
from pathlib import Path

from Bm25 import normalize
from rag import retrieve

KS = (1, 3, 5)
MODES = ("vector", "bm25", "hybrid")


def load_cases() -> list[dict]:
    path = Path("eval_set.json")
    if not path.exists():
        raise SystemExit(
            "eval_set.json introuvable : copie eval_set.example.json en eval_set.json "
            "puis adapte-le à tes documents (docs/)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_query(case: dict) -> str:
    """Texte de recherche : la 'question' du cas, ou à défaut celui construit depuis le profil."""
    if case.get("question"):
        return case["question"]
    from main import StartupProfile, build_query

    return build_query(StartupProfile(**case["profil"]))


def is_relevant(chunk: dict, case: dict) -> bool:
    """Un chunk est pertinent s'il vient d'un fichier attendu ET contient les termes attendus."""
    if chunk["source"] not in case["expected_sources"]:
        return False
    text = normalize(chunk["text"])
    return all(normalize(t) in text for t in case.get("must_contain", []))


# ---------------------------------------------------------------------------
# 1. Évaluation de la RECHERCHE
# ---------------------------------------------------------------------------
def eval_retrieval(cases: list[dict]) -> dict:
    results = {}
    for mode in MODES:
        hits = {k: 0 for k in KS}
        reciprocal_ranks, failed = 0.0, []
        for case in cases:
            chunks = retrieve(get_query(case), k=max(KS), mode=mode)
            rank = next((i for i, c in enumerate(chunks, 1) if is_relevant(c, case)), None)
            for k in KS:
                if rank is not None and rank <= k:
                    hits[k] += 1
            reciprocal_ranks += 1 / rank if rank else 0.0
            if rank is None:
                failed.append(
                    {
                        "question": get_query(case)[:80],
                        "attendu": case["expected_sources"],
                        "trouve": [c["source"] for c in chunks],
                    }
                )
        n = len(cases)
        results[mode] = {
            **{f"recall@{k}": round(hits[k] / n, 3) for k in KS},
            "MRR": round(reciprocal_ranks / n, 3),
            "echecs_top5": failed,
        }
    return results


def print_retrieval(results: dict, n: int) -> None:
    print(f"\n=== RECHERCHE ({n} cas de test) ===")
    print(f"{'mode':<8}" + "".join(f"{'recall@' + str(k):>10}" for k in KS) + f"{'MRR':>8}")
    for mode, r in results.items():
        print(f"{mode:<8}" + "".join(f"{r[f'recall@{k}']:>10}" for k in KS) + f"{r['MRR']:>8}")
    for mode in MODES:
        for f in results[mode]["echecs_top5"]:
            print(f"  [{mode}] raté : {f['question']!r} | attendu {f['attendu']} | trouvé {f['trouve']}")


# ---------------------------------------------------------------------------
# 2. Évaluation de la FIDÉLITÉ (les réponses sont-elles appuyées par les extraits ?)
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """Tu es un vérificateur strict. On te donne des EXTRAITS de documents et une
AFFIRMATION (une réglementation avec sa justification et l'action recommandée).
Réponds UNIQUEMENT en JSON : {"supported": true ou false, "reason": "une phrase"}.
supported = true SEULEMENT si l'affirmation est clairement appuyée par les EXTRAITS fournis.
Si l'affirmation ajoute des éléments absents des extraits, réponds false."""


def judge(claim: str, extraits_text: str) -> tuple[bool, str]:
    from main import MODEL, client

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"EXTRAITS:\n{extraits_text}\n\nAFFIRMATION:\n{claim}"},
            ],
            response_format={"type": "json_object"},
        )
        verdict = json.loads(resp.choices[0].message.content)
        return bool(verdict.get("supported")), str(verdict.get("reason", ""))
    except Exception as e:  # le juge peut échouer (quota, JSON invalide) : on compte non fidèle
        return False, f"erreur du juge : {e}"


def eval_faithfulness(cases: list[dict]) -> dict:
    from main import StartupProfile, run_agent

    total = sourced = general = citation_ok = supported = 0
    details = []
    for case in [c for c in cases if c.get("profil")]:
        rapport, extraits = run_agent(StartupProfile(**case["profil"]))
        for r in rapport.reglementations:
            total += 1
            if r.source.lower().startswith("connaissance"):
                general += 1  # non vérifiable avec les extraits : on le compte à part
                continue
            sourced += 1
            cited = [e for e in extraits if e["source"] == r.source]
            if cited:
                citation_ok += 1  # le fichier cité fait bien partie des extraits fournis
                ok, reason = judge(
                    f"{r.nom} : {r.pourquoi} Action : {r.action}",
                    "\n\n".join(e["text"] for e in cited),
                )
            else:
                ok, reason = False, "source citée absente des extraits fournis"
            supported += ok
            details.append({"reglementation": r.nom, "source": r.source, "fidele": ok, "raison": reason})

    return {
        "reglementations_total": total,
        "part_connaissance_generale": round(general / total, 3) if total else None,
        "citations_valides": round(citation_ok / sourced, 3) if sourced else None,
        "fidelite": round(supported / sourced, 3) if sourced else None,
        "details": details,
    }


def print_faithfulness(r: dict) -> None:
    print("\n=== FIDÉLITÉ DES RÉPONSES ===")
    print(f"Réglementations générées        : {r['reglementations_total']}")
    print(f"Part 'connaissance générale'    : {r['part_connaissance_generale']}  (non vérifiable)")
    print(f"Citations valides (fichier OK)  : {r['citations_valides']}")
    print(f"Fidélité (appuyé par l'extrait) : {r['fidelite']}")
    for d in r["details"]:
        if not d["fidele"]:
            print(f"  non fidèle : {d['reglementation']} [{d['source']}] -> {d['raison']}")


if __name__ == "__main__":
    cases = load_cases()
    report = {"retrieval": eval_retrieval(cases)}
    print_retrieval(report["retrieval"], len(cases))
    if "--faithfulness" in sys.argv:
        report["faithfulness"] = eval_faithfulness(cases)
        print_faithfulness(report["faithfulness"])
    Path("eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRapport détaillé : eval_report.json")