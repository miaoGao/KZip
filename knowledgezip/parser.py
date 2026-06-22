"""Parse the LLM output of the compression stage into triples with citations."""

import re


def parse_triples(text):
    """Extract triples of the form

        - (`head`, `relation`, `tail`) [cite: source sentence]

    from `text`. Returns a list of dicts: {"triple": [h, r, t], "cite": str}.
    """
    if not text:
        return []

    # Only look at the triples section if the bold header is present.
    m = re.search(r"\*\*Relevant Triples:\*\*(.*)", text, re.DOTALL)
    if m:
        text = m.group(1)

    triples = []
    for line in re.findall(r"^[ \t]*-[ \t]*(.*)$", text, re.MULTILINE):
        line = line.strip()
        if not line:
            continue
        match = re.search(r"\((.*)\).+?\[cite:(.*)\]", line)
        if not match:
            continue
        elements = match.group(1).strip().split("`, `")
        if len(elements) != 3:
            continue
        head = elements[0].strip("`")
        relation = elements[1].strip("`")
        tail = elements[2].strip("`")
        triples.append({"triple": [head, relation, tail], "cite": match.group(2).strip()})
    return triples


def triples_to_str(triples):
    return "\n".join(
        f"- (`{h}`, `{r}`, `{t}`) [cite: {item['cite']}]"
        for item in triples
        for h, r, t in [item["triple"]]
    )
