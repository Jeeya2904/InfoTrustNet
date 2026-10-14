"""Token-occlusion importance and readable model explanations."""
import numpy as np
def occlusion_importance(fused_fn, text, source="", top_k=6):
    tokens = text.split()
    if not tokens: return []
    variants = [" ".join(tokens[:i] + tokens[i + 1:]) for i in range(len(tokens))]
    base = fused_fn([text], [source])[0]
    probs = fused_fn(variants, [source] * len(variants))
    deltas = base - probs
    order = np.argsort(-np.abs(deltas))[:top_k]
    return [(tokens[i], float(deltas[i])) for i in order]
def narrative(pred, kg, strategy, contributions, tokens):
    verdict = "likely MISINFORMATION" if pred["is_misinformation"] else "likely RELIABLE"
    lines = [f"Verdict: {verdict} (P(misinformation) = {pred['p_misinformation']:.2f})."]
    if kg.contradict: lines.append("Knowledge graph conflicts with trusted facts: " + "; ".join(kg.contradict) + ".")
    elif kg.support: lines.append("Knowledge graph matches trusted facts: " + "; ".join(kg.support) + ".")
    else: lines.append("Knowledge graph has no matching facts (neutral evidence).")
    top = max(contributions, key=lambda k: abs(contributions[k]))
    lines.append(f"RL policy selected '{strategy}'; strongest signal: {top} ({contributions[top]:+.2f} logit).")
    if tokens: lines.append("Most influential words: " + ", ".join(f"'{t}' ({d:+.2f})" for t, d in tokens[:4]) + ".")
    return "\n".join(lines)
