### Retrieval (42 answerable questions)

| mode | hit@1 | hit@3 | hit@5 | MRR@5 |
|---|---|---|---|---|
| dense | 64% | 90% | 93% | 0.76 |
| bm25 | 71% | 90% | 95% | 0.82 |
| hybrid | 76% | 95% | 98% | 0.85 |
| rerank | 83% | 95% | 100% | 0.90 |

Hybrid misses (not in top 5):
- For which kinds of AST methods can essential agreement be used?

### Refusal threshold (score = similarity, mode = hybrid)

| threshold | unanswerable refused | answerable wrongly refused |
|---|---|---|
| 0.583 | 0% | 0% |
| 0.609 | 0% | 2% |
| 0.635 | 7% | 2% |
| 0.661 | 21% | 5% |
| 0.687 | 29% | 14% |
| 0.713 | 43% | 29% |
| 0.739 | 57% | 38% |
| 0.765 | 71% | 62% |
| 0.791 | 93% | 81% |
| 0.817 | 93% | 88% |
| 0.843 | 100% | 98% |

Separation AUC: **0.61** (0.5 = useless, 1.0 = perfect). Mean best score: answerable 0.75, unanswerable 0.72. Best balanced threshold: **0.739**
