"""
biblio_figures.py — Generator figur bibliometrik / science-mapping (SLNA) untuk SLR.

Menggantikan langkah manual VOSviewer/biblioshiny dengan generator TER-SKRIP &
DETERMINISTIK, sehingga figur + data mentahnya bisa di-arsip (Zenodo) dan
DIREPRODUKSI pihak ketiga — menutup rantai dokumentasi (bukan screenshot manual).

Sumber data: MongoDB `slr_agentic_db` (koleksi `slr_extraction` = studi included final,
di-enrich metadata dari `slr_screening`). READ-ONLY. Kredensial dari ENV (MONGO_URI,
DB_NAME) — tidak pernah di-hardcode.

Output per sesi (folder --out):
  - annual_production.{png,svg,pdf}          + annual_production.csv
  - top_sources.{png,svg,pdf}                + top_sources.csv
  - keyword_cooccurrence.{png,svg,pdf}       + keyword_edges.csv, keyword_nodes.csv
  - thematic_map.{png,svg,pdf}               + thematic_map.csv
  - collaboration_network.{png,svg,pdf}      + collaboration_edges.csv
  - manifest.json  (parameter + daftar artefak + hitungan — untuk reproducibility)

Semua deterministik (sort stabil + layout seed tetap) → run ulang = figur identik.

CLI:  python -m core.biblio_figures --session <session_id> --out ./data/figures/<session_id>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

# Backend non-interaktif (Colab/headless): set SEBELUM import pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv opsional
    pass

LAYOUT_SEED = 42  # determinisme layout jaringan
_FORMATS = ("png", "svg", "pdf")

# Stopword generik agar keyword-map tak didominasi kata umum (multi-tenant: minimal, netral).
_STOPWORDS = {
    "study", "studies", "analysis", "approach", "method", "methods", "based",
    "using", "use", "used", "review", "systematic", "paper", "research",
    "results", "data", "model", "models", "effect", "effects", "application",
    "applications", "case", "system", "systems", "new", "novel", "via",
}


# ─────────────────────────────── util ───────────────────────────────

def _get(d: dict, *keys, default=""):
    """Ambil nilai pertama yang ada dari beberapa varian nama field (case/spasi)."""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    # coba case-insensitive
    low = {str(k).lower().replace(" ", "_"): v for k, v in d.items()}
    for k in keys:
        kk = str(k).lower().replace(" ", "_")
        if kk in low and low[kk] not in (None, ""):
            return low[kk]
    return default


def _year_of(v) -> int | None:
    if v in (None, ""):
        return None
    m = re.search(r"(19|20)\d{2}", str(v))
    return int(m.group(0)) if m else None


def _split_multi(s: str) -> list[str]:
    """Pisah string multi-nilai (keyword/author) atas ; , | dan bersihkan."""
    if not s:
        return []
    parts = re.split(r"[;,|/]", str(s))
    out = []
    for p in parts:
        t = p.strip()
        if t:
            out.append(t)
    return out


def _norm_kw(kw: str) -> str:
    t = re.sub(r"\s+", " ", kw.strip().lower())
    return t


def _save(fig, outdir: str, name: str) -> list[str]:
    paths = []
    for fmt in _FORMATS:
        p = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(os.path.basename(p))
    plt.close(fig)
    return paths


def _write_csv(outdir: str, name: str, header: list[str], rows: list[list]) -> str:
    import csv
    p = os.path.join(outdir, f"{name}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return os.path.basename(p)


# ─────────────────────────────── data ───────────────────────────────

def load_corpus(session_id: str, mongo_uri: str = "", db_name: str = "",
                use_all: bool = False) -> list[dict]:
    """Muat korpus studi INCLUDED (dari slr_extraction) + enrich metadata dari slr_screening.

    Kembalikan list of dict ternormalisasi: {title, authors[], keywords[], year, source, doi, cited}.
    """
    from pymongo import MongoClient

    mongo_uri = mongo_uri or os.environ.get("MONGO_URI", "")
    db_name = db_name or os.environ.get("DB_NAME", "slr_agentic_db")
    if not mongo_uri:
        raise SystemExit("MONGO_URI tak diset (ENV/.env). Wajib untuk membaca korpus.")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=20000)
    db = client[db_name]

    # 1) Set DOI/title studi included final = yang ada di slr_extraction.
    included_dois, included_titles = set(), set()
    for e in db["slr_extraction"].find({"session_id": session_id}, {"DOI": 1, "Title": 1, "doi": 1, "title": 1}):
        doi = str(_get(e, "DOI", "doi")).lower().strip()
        ttl = str(_get(e, "Title", "title")).lower().strip()
        if doi:
            included_dois.add(doi)
        if ttl:
            included_titles.add(ttl)

    # 2) Ambil metadata bibliografis dari slr_screening.
    screening = list(db["slr_screening"].find({"session_id": session_id}))
    client.close()

    def is_included(p) -> bool:
        if use_all:
            return True
        doi = str(_get(p, "DOI", "doi")).lower().strip()
        ttl = str(_get(p, "Title", "title")).lower().strip()
        if included_dois or included_titles:
            return (doi and doi in included_dois) or (ttl and ttl in included_titles)
        # fallback: filter inklusi ala nsa (Final_Decision / Screener_1_Decision)
        fd = str(_get(p, "Final_Decision", "final_decision"))
        s1 = str(_get(p, "Screener_1_Decision", "screener_1_decision"))
        return fd == "INCLUDE" or (fd == "" and s1 == "INCLUDE")

    corpus = []
    for p in screening:
        if not is_included(p):
            continue
        corpus.append({
            "title": str(_get(p, "Title", "title")),
            "authors": _split_multi(_get(p, "Authors", "authors", "Author", "author")),
            "keywords": [_norm_kw(k) for k in _split_multi(
                _get(p, "Keywords", "keywords", "Author Keywords", "Index Keywords"))],
            "year": _year_of(_get(p, "Year", "year", "Publication Year", "Date")),
            "source": str(_get(p, "Source", "source", "Journal", "journal",
                               "Publication", "Source title", "Source_title")).strip(),
            "doi": str(_get(p, "DOI", "doi")).strip(),
            "cited": 0,
        })
        c = _get(p, "Cited", "cited", "Cited by", "Citations", "citations", default="")
        try:
            corpus[-1]["cited"] = int(re.sub(r"[^\d]", "", str(c)) or 0)
        except Exception:
            corpus[-1]["cited"] = 0
    return corpus


# ─────────────────────────────── figur ───────────────────────────────

def fig_annual_production(corpus, outdir) -> dict:
    years = [p["year"] for p in corpus if p["year"]]
    if not years:
        return {}
    lo, hi = min(years), max(years)
    counts = Counter(years)
    xs = list(range(lo, hi + 1))
    ys = [counts.get(y, 0) for y in xs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", color="#0e7c7b", linewidth=2)
    ax.fill_between(xs, ys, color="#0e7c7b", alpha=0.12)
    ax.set_title("Annual Scientific Production", fontsize=13, weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Documents")
    ax.grid(True, alpha=0.25)
    ax.set_xticks(xs)
    ax.tick_params(axis="x", rotation=45)
    files = _save(fig, outdir, "annual_production")
    csv = _write_csv(outdir, "annual_production", ["year", "documents"], [[y, counts.get(y, 0)] for y in xs])
    return {"figure": "annual_production", "files": files, "data": csv, "n_years": len(xs)}


def fig_top_sources(corpus, outdir, top_n=15) -> dict:
    srcs = Counter(p["source"] for p in corpus if p["source"])
    if not srcs:
        return {}
    items = sorted(srcs.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    labels = [s if len(s) <= 45 else s[:42] + "…" for s, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(items) + 1)))
    ax.barh(labels, vals, color="#b45309")
    ax.set_title("Most Relevant Sources", fontsize=13, weight="bold")
    ax.set_xlabel("Documents")
    ax.grid(True, axis="x", alpha=0.25)
    files = _save(fig, outdir, "top_sources")
    csv = _write_csv(outdir, "top_sources", ["source", "documents"],
                     [[s, v] for s, v in sorted(srcs.items(), key=lambda kv: (-kv[1], kv[0]))])
    return {"figure": "top_sources", "files": files, "data": csv, "n_sources": len(srcs)}


def _keyword_graph(corpus, min_freq=2, min_edge=1):
    freq = Counter()
    co = Counter()
    for p in corpus:
        kws = sorted(set(k for k in p["keywords"] if len(k) > 2 and k not in _STOPWORDS))
        for k in kws:
            freq[k] += 1
        for a, b in combinations(kws, 2):
            co[(a, b)] += 1
    keep = {k for k, v in freq.items() if v >= min_freq}
    G = nx.Graph()
    for k in sorted(keep):
        G.add_node(k, occ=freq[k])
    for (a, b), w in sorted(co.items()):
        if w >= min_edge and a in keep and b in keep:
            G.add_edge(a, b, weight=w)
    return G, freq


def fig_keyword_cooccurrence(corpus, outdir, min_freq=2) -> dict:
    G, freq = _keyword_graph(corpus, min_freq=min_freq)
    if G.number_of_nodes() == 0:
        return {"figure": "keyword_cooccurrence", "skipped": "keyword terlalu jarang (naikkan cakupan / metadata keyword kosong)"}
    # komunitas deterministik untuk pewarnaan
    comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    color_of = {}
    palette = ["#0e7c7b", "#b45309", "#4338ca", "#be123c", "#15803d", "#7c3aed", "#0369a1", "#a16207"]
    for i, c in enumerate(comms):
        for n in c:
            color_of[n] = palette[i % len(palette)]
    pos = nx.spring_layout(G, seed=LAYOUT_SEED, weight="weight", k=0.5)
    fig, ax = plt.subplots(figsize=(10, 8))
    sizes = [80 + 40 * G.nodes[n]["occ"] for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=[0.3 * G[u][v]["weight"] for u, v in G.edges()], ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=[color_of[n] for n in G.nodes()], alpha=0.85, ax=ax)
    # label hanya node ber-occ tinggi agar tak penuh
    thr = sorted((G.nodes[n]["occ"] for n in G.nodes()), reverse=True)[:min(30, G.number_of_nodes())][-1]
    labels = {n: n for n in G.nodes() if G.nodes[n]["occ"] >= thr}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title("Keyword Co-occurrence Network", fontsize=13, weight="bold")
    ax.axis("off")
    files = _save(fig, outdir, "keyword_cooccurrence")
    edges_csv = _write_csv(outdir, "keyword_edges", ["keyword_a", "keyword_b", "cooccurrence"],
                           [[u, v, G[u][v]["weight"]] for u, v in sorted(G.edges())])
    nodes_csv = _write_csv(outdir, "keyword_nodes", ["keyword", "occurrences", "cluster"],
                           [[n, G.nodes[n]["occ"],
                             next((i for i, c in enumerate(comms) if n in c), -1)]
                            for n in sorted(G.nodes())])
    return {"figure": "keyword_cooccurrence", "files": files, "data": [edges_csv, nodes_csv],
            "n_keywords": G.number_of_nodes(), "n_edges": G.number_of_edges(), "n_clusters": len(comms)}


def fig_thematic_map(corpus, outdir, min_freq=2) -> dict:
    """Thematic map (Callon centrality × density) — figur khas SLNA/biblioshiny."""
    G, freq = _keyword_graph(corpus, min_freq=min_freq)
    if G.number_of_edges() == 0:
        return {"figure": "thematic_map", "skipped": "jaringan keyword kosong"}
    comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    rows = []
    for c in comms:
        cset = set(c)
        n = len(cset)
        internal = sum(G[u][v]["weight"] for u, v in G.edges() if u in cset and v in cset)
        external = sum(G[u][v]["weight"] for u, v in G.edges()
                       if (u in cset) ^ (v in cset))
        occ = sum(G.nodes[x]["occ"] for x in cset)
        density = (internal / n) * 10 if n else 0.0
        centrality = external * 10
        top = sorted(cset, key=lambda x: (-G.nodes[x]["occ"], x))[:3]
        rows.append({"label": ", ".join(top), "centrality": centrality,
                     "density": density, "occ": occ, "size": n})
    if not rows:
        return {"figure": "thematic_map", "skipped": "tak ada cluster"}
    cx = sorted(r["centrality"] for r in rows)
    cy = sorted(r["density"] for r in rows)
    mx = cx[len(cx) // 2]
    my = cy[len(cy) // 2]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.axvline(mx, color="#999", linestyle="--", linewidth=1)
    ax.axhline(my, color="#999", linestyle="--", linewidth=1)
    maxocc = max(r["occ"] for r in rows) or 1
    for r in rows:
        ax.scatter(r["centrality"], r["density"], s=200 + 1500 * r["occ"] / maxocc,
                   color="#0e7c7b", alpha=0.45, edgecolors="#0e7c7b")
        ax.annotate(r["label"], (r["centrality"], r["density"]), fontsize=8,
                    ha="center", va="center")
    ax.margins(0.18)  # ruang agar label cluster di tepi tak terpotong
    # label kuadran
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    ax.text(xhi, yhi, "Motor themes", ha="right", va="top", fontsize=9, color="#555", style="italic")
    ax.text(xlo, yhi, "Niche themes", ha="left", va="top", fontsize=9, color="#555", style="italic")
    ax.text(xhi, ylo, "Basic themes", ha="right", va="bottom", fontsize=9, color="#555", style="italic")
    ax.text(xlo, ylo, "Emerging/declining", ha="left", va="bottom", fontsize=9, color="#555", style="italic")
    ax.set_title("Thematic Map (Callon centrality × density)", fontsize=13, weight="bold")
    ax.set_xlabel("Centrality (relevance)")
    ax.set_ylabel("Density (development)")
    files = _save(fig, outdir, "thematic_map")
    csv = _write_csv(outdir, "thematic_map", ["cluster_top_terms", "centrality", "density", "occurrences", "n_keywords"],
                     [[r["label"], round(r["centrality"], 3), round(r["density"], 3), r["occ"], r["size"]] for r in rows])
    return {"figure": "thematic_map", "files": files, "data": csv, "n_clusters": len(rows)}


def fig_collaboration(corpus, outdir, min_freq=2, top_authors=60) -> dict:
    freq = Counter()
    co = Counter()
    for p in corpus:
        auth = sorted(set(a for a in p["authors"] if a))
        for a in auth:
            freq[a] += 1
        for a, b in combinations(auth, 2):
            co[(a, b)] += 1
    if not co:
        return {"figure": "collaboration_network", "skipped": "metadata penulis kosong / tak ada ko-penulisan"}
    keep = {a for a, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:top_authors]}
    G = nx.Graph()
    for a in sorted(keep):
        G.add_node(a, docs=freq[a])
    for (a, b), w in sorted(co.items()):
        if a in keep and b in keep:
            G.add_edge(a, b, weight=w)
    if G.number_of_edges() == 0:
        return {"figure": "collaboration_network", "skipped": "tak ada edge ko-penulisan di antara top authors"}
    comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    palette = ["#0e7c7b", "#b45309", "#4338ca", "#be123c", "#15803d", "#7c3aed"]
    color_of = {}
    for i, c in enumerate(comms):
        for n in c:
            color_of[n] = palette[i % len(palette)]
    pos = nx.spring_layout(G, seed=LAYOUT_SEED, weight="weight", k=0.6)
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=[80 + 60 * G.nodes[n]["docs"] for n in G.nodes()],
                           node_color=[color_of.get(n, "#0e7c7b") for n in G.nodes()], alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title("Author Collaboration Network", fontsize=13, weight="bold")
    ax.axis("off")
    files = _save(fig, outdir, "collaboration_network")
    csv = _write_csv(outdir, "collaboration_edges", ["author_a", "author_b", "co_authored"],
                     [[u, v, G[u][v]["weight"]] for u, v in sorted(G.edges())])
    return {"figure": "collaboration_network", "files": files, "data": csv,
            "n_authors": G.number_of_nodes(), "n_edges": G.number_of_edges()}


# ─────────────────────────────── orkestrasi ───────────────────────────────

def generate_all(session_id: str, outdir: str, mongo_uri: str = "", db_name: str = "",
                 min_freq: int = 2, use_all: bool = False) -> dict:
    os.makedirs(outdir, exist_ok=True)
    corpus = load_corpus(session_id, mongo_uri, db_name, use_all=use_all)
    manifest = {
        "session_id": session_id,
        "corpus_size": len(corpus),
        "min_keyword_freq": min_freq,
        "use_all": use_all,
        "generator": "pede/core/biblio_figures.py",
        "layout_seed": LAYOUT_SEED,
        "note": "Figur deterministik; run ulang menghasilkan hasil identik. Sertakan folder ini + CSV di deposit Zenodo untuk reproducibility.",
        "figures": [],
    }
    if not corpus:
        manifest["error"] = "Korpus kosong (tak ada studi included / metadata). Pastikan Modul 7 selesai."
        with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return manifest

    for fn in (fig_annual_production, fig_top_sources):
        r = fn(corpus, outdir)
        if r:
            manifest["figures"].append(r)
    for fn in (fig_keyword_cooccurrence, fig_thematic_map):
        manifest["figures"].append(fn(corpus, outdir, min_freq=min_freq))
    manifest["figures"].append(fig_collaboration(corpus, outdir, min_freq=min_freq))

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Generator figur bibliometrik/SLNA (deterministik) dari MongoDB SLR.")
    ap.add_argument("--session", required=True, help="session_id SLR (slr_sessions._id)")
    ap.add_argument("--out", default="", help="folder output (default ./data/figures/<session>)")
    ap.add_argument("--mongo-uri", default="", help="override MONGO_URI (default dari ENV/.env)")
    ap.add_argument("--db", default="", help="override DB_NAME (default slr_agentic_db)")
    ap.add_argument("--min-freq", type=int, default=2, help="ambang minimal frekuensi keyword/penulis")
    ap.add_argument("--all", action="store_true", help="pakai SEMUA paper screening (bukan hanya included)")
    args = ap.parse_args()
    outdir = args.out or os.path.join("data", "figures", args.session)
    manifest = generate_all(args.session, outdir, args.mongo_uri, args.db, args.min_freq, args.all)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    ok = sum(1 for f in manifest.get("figures", []) if f.get("files"))
    print(f"\n[biblio_figures] {ok} figur ditulis ke {outdir} (korpus {manifest.get('corpus_size', 0)} studi).")


if __name__ == "__main__":
    main()
