from collections import defaultdict

def chunk_key(item: dict) -> tuple:
    
    return (
        item["source"],
        item["section"],
        item["chunk_index"],
    )

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    *,
    k: int = 60,
    limit: int = 20,
) -> list[dict]:
    
    fused_scores: dict[tuple, float] = defaultdict(float)
    candidates: dict[tuple, dict] = {}
    
    for results in ranked_lists:
        
        for rank, result in enumerate(results, start=1):
            
            key = chunk_key(result)
            fused_scores[key] += 1.0 / (k+rank)
            
            candidates.setdefault(key, result)
            
    ordered_keys = sorted(
        
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )
    
    fused = []
    for key in ordered_keys[:limit]:
        
        result = dict(candidates[key])
        result["fusion_score"] = fused_scores[key]
        
        fused.append(result)
        
    return fused

