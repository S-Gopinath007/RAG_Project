### Whole-system threshold sweep (score = rerank_score, mode = rerank, model = qwen/qwen3.8-27b)

| threshold | unanswerable refused | answerable wrongly refused | correct citation (of answered) |
|---|---|---|---|
| none (LLM check only) | 14/14 | 6/42 (14%) | 36/36 |
| -6 | 14/14 | 7/42 (17%) | 35/35 |
| -4 | 14/14 | 8/42 (19%) | 34/34 |
| -3 | 14/14 | 8/42 (19%) | 34/34 |
| -2 | 14/14 | 9/42 (21%) | 33/33 |
| -1 | 14/14 | 10/42 (24%) | 32/32 |
| 0 | 14/14 | 11/42 (26%) | 31/31 |
| 0.3 | 14/14 | 11/42 (26%) | 31/31 |
| 1 | 14/14 | 13/42 (31%) | 29/29 |
| 2 | 14/14 | 17/42 (40%) | 25/25 |
| 4 | 14/14 | 20/42 (48%) | 22/22 |

Lowest setting that still refuses every unanswerable question: **no threshold at all** (wrongly refuses 6/42 answerable).
