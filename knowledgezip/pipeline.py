"""The KnowledgeZip inference pipeline: compress -> (select) -> answer."""

from . import reward
from .parser import parse_triples, triples_to_str
from .prompts import ANSWER_PROMPT, EXTRACT_PROMPT


def build_context(question, gt_only=False):
    """Render the supporting documents of a question into a single string."""
    docs = question["context"]
    if gt_only:
        docs = [d for d in docs if d.get("is_supporting")]
    body = "".join(f"#### Document {i + 1}:\n{d['content']}\n\n" for i, d in enumerate(docs))
    return f"### Supporting Documents:\n{body.strip()}"


def compress(llm, question, context, num_samples=1):
    """Stage 1: extract triples-with-citations from the documents.

    When `num_samples > 1`, draw several candidate graphs (with temperature) and
    keep the one with the highest reward (Stage 2: select). With a single sample
    we just greedily decode.
    """
    if num_samples <= 1:
        output = llm.complete(EXTRACT_PROMPT.format(context=context, question=question["question"]), temperature=0.0)
        return parse_triples(output)

    candidates = []
    for _ in range(num_samples):
        output = llm.complete(EXTRACT_PROMPT.format(context=context, question=question["question"]), temperature=0.7)
        triples = parse_triples(output)
        if triples:
            candidates.append(triples)
    if not candidates:
        return []
    return max(candidates, key=lambda t: reward.total(t, context, question["question"]))


def answer(llm, question, triples):
    """Stage 3: reason over the compressed graph to produce the final answer."""
    prompt = ANSWER_PROMPT.format(triples=triples_to_str(triples), question=question["question"])
    output = llm.complete(prompt, temperature=0.0)
    pred = output.split("**Final Answer:**")[-1].strip()
    return output, pred


def run_question(llm, question, num_samples=1, gt_only=False):
    context = build_context(question, gt_only=gt_only)
    triples = compress(llm, question, context, num_samples=num_samples)
    llm_output, pred = answer(llm, question, triples)
    return {
        "id": question["id"],
        "question": question["question"],
        "answer": question["answer"],
        "triples": triples,
        "llm_output": llm_output,
        "pred_ans": pred,
    }
