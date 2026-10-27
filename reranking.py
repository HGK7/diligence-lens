from functools import lru_cache

from sentence_transformers import CrossEncoder

@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    )
    
    

def rerank(
    question: str,
    candidates: list[dict],
    *,
    limit: int = 6,
) -> list[dict]:
    
    if not candidates:
        return []
    
    pairs = [
        (
            
            question,
            " ".join(
                [
                    candidate.get("section", ""),
                    candidate["text"],
                ]
            ),
        )
        for candidate in candidates
    ]
    
    scores = get_reranker().predict(pairs)
    ranked = []
    
    for candidate, score in zip(candidates, scores, strict=True):
        
        item = dict(candidate)
        item["rerank_score"] = float(score)
        
        ranked.append(item)
        
    ranked.sort(key = lambda item: item["rerank_score"], reverse= True)
    return ranked[:limit]

