from collections import Counter

from lexical_retrieval import lexical_tokens

def diversify_by_source(
    ranked: list[dict],
    *,
    limit: int = 8,
    max_per_source: int = 2,
) -> list[dict]:
    
    selected: list[dict] = []
    source_counts: Counter[str] = Counter()
    
    for item in ranked:
        source = item["source"]
        if source_counts[source] >= max_per_source:
            
            continue
        
        selected.append(item)
        source_counts[source] += 1
        
        if len(selected) == limit:
            
            break
        
    return selected

def jaccard_similarity(left: str, right: str) ->float: 
    
    right_tokens = set(lexical_tokens(right))
    left_tokens = set(lexical_tokens(left))
    
    union = left_tokens | right_tokens
    
    if not union:
        
        return 0.0
    return len(left_tokens & right_tokens) / len(union)