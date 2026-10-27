# Evaluation cases

`retrieval_cases.json` — labeled questions for Diligence Lens (~30).

| Field | Meaning |
|-------|---------|
| `category` | `hyper_specific`, `semantic_vague`, `hard_distractor`, `corpus_gap`, `nonsense`, `prompt_injection`, `stupid` |
| `expect_hit` | `true` → scored in Recall/MRR against `expected_sources`. `false` → boundary case (abstain / ignore injection); **excluded** from Recall/MRR averages |
| `expected_decision` | Hint for review/Quality-lab style checks (`insufficient_evidence`, etc.) |

Frozen Part 1–8 metrics used the original 10 `expect_hit=true` cases. Re-baseline after pool expansion if you compare to old numbers.

```powershell
python scripts/probe_score_distributions.py --all --threshold 4 --final-limit 8
python scripts/evaluate_rerank.py
```
