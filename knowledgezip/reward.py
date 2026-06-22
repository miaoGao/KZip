"""Reference-free reward for selecting the best compressed graph.

This is a minimal, dependency-light version of the reward used in the paper. It
scores a set of extracted triples along several axes that correlate with a
useful, faithful compression, without needing the gold answer:

  * fmt          - fraction of well-formed triples
  * triple_faith - head & tail actually appear in the cited sentence
  * cite_faith   - the cited sentence actually appears in the source documents
  * uniqueness   - triples / citations are not redundant
  * logic_path   - question entities connect to each other in the triple graph
  * num_triples  - prefer a small, focused set (2-4 triples)
  * relevance    - lexical overlap between the triples and the question

The total score is the sum of the components. Higher is better.
"""

import networkx as nx

from .eval import normalize


def score(triples, context, question):
    s = {
        "fmt": 0.0,
        "triple_faith": 0.0,
        "cite_faith": 0.0,
        "uniqueness": 0.0,
        "logic_path": 0.0,
        "num_triples": 0.0,
        "relevance": 0.0,
    }
    if not triples:
        s["fmt"] = -1.0
        return s

    s["fmt"] = 1.0
    norm_context = normalize(context)

    g = nx.Graph()
    norm_triples, norm_cites, triple_words = [], [], []
    for item in triples:
        h, r, t = item["triple"]
        nh, nr, nt = normalize(h), normalize(r), normalize(t)
        ncite = normalize(item["cite"])
        norm_cites.append(ncite)
        if nh and nt:
            g.add_edge(nh, nt)
            norm_triples.append((nh, nr, nt))
        triple_words += nh.split() + nr.split() + nt.split()

        if nh and nt and nh in ncite and nt in ncite:
            s["triple_faith"] += 1.0
        if ncite and ncite in norm_context:
            s["cite_faith"] += 1.0

    n = len(triples)
    s["triple_faith"] /= n
    s["cite_faith"] /= n
    if norm_triples:
        s["uniqueness"] = len(set(norm_triples)) / len(norm_triples)

    # logic path: do any two question-entities connect through the triple graph?
    q_words = set(normalize(question).split())
    q_nodes = [node for node in g.nodes if q_words & set(node.split())]
    if len(q_nodes) >= 2:
        connected = any(
            nx.has_path(g, a, b)
            for i, a in enumerate(q_nodes)
            for b in q_nodes[i + 1:]
        )
        s["logic_path"] = 1.0 if connected else 0.0

    if 2 <= n <= 4:
        s["num_triples"] = 1.0
    elif n > 4:
        s["num_triples"] = max(-1.0, 1.0 - (n - 4) * 0.5)

    triple_set = set(triple_words)
    if q_words | triple_set:
        s["relevance"] = len(q_words & triple_set) / len(q_words | triple_set)

    return s


def total(triples, context, question):
    return sum(score(triples, context, question).values())
