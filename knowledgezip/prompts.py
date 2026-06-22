"""Prompts for the two inference stages of KnowledgeZip.

Stage 1 (Compress): read the supporting documents and extract the few triples
that are relevant to the question, each grounded by a citation to its source
sentence.

Stage 2 (Answer): reason over the compressed triples only and produce a short
final answer.
"""

EXTRACT_PROMPT = """
{context}

### Question:
{question}

### Instruction:
Analyze what information is needed to answer the question, then extract the
triples (head, relation, tail) from the documents above that are most relevant.

### Requirements:
1. **Reasoning Process:** Briefly analyze the question and identify which documents contain the key information.
2. Each triple must be in the format (`head`, `relation`, `tail`). Each element must be wrapped in backticks.
3. Only include information directly relevant to the question.
4. The extracted triples must be sufficient to answer the question.
5. Output as a list, each triple starting with ' - '.
6. For each triple, append a citation `[cite: source sentence]` quoting the original sentence it came from.
7. Use the EXACT bold headers below. Do NOT use Markdown headers like #, ##, or ###.

### Triple Example:
- (`Albert Einstein`, `born in`, `Ulm`) [cite: Albert Einstein was born in Ulm, in the German Empire, on 14 March 1879.]

### Response Format:
**Reasoning Process:**
<your analysis>

**Relevant Triples:**
- (`head`, `relation`, `tail`) [cite: source sentence]
- (`head`, `relation`, `tail`) [cite: source sentence]
...
""".strip()


ANSWER_PROMPT = """
### Knowledge Graph (head, relation, tail) [cite: source]:
{triples}

### Question:
{question}

### Instruction:
Based only on the knowledge graph above, answer the question. Do not rely on
external knowledge that is not supported by the provided triples.

### Response Format:
**Reasoning Process:**
<one sentence reasoning>

**Final Answer:**
<answer without explanation in a few words>
""".strip()
