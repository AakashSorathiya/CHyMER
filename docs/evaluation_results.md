## NLI inference results with generic hypotheses

| Model | TP | TN | FP | FN | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeBERTa-v3-base-mnli-fever-anli | 389 | 221 | 741 | 25 | 0.34 | 0.93 | 0.50 |
| t5-base | 0 | 962 | 0 | 414 | 0 | 0 | 0 |
| nli-roberta-base | 398 | 120 | 842 | 16 | 0.32 | 0.96 | 0.48
| roberta-large-mnli | 335 | 365 | 597 | 79 | 0.35 | 0.80 | 0.49

## Comparison of domain-specific and generic hypotheses using best NLI model

| Hypotheses | TP | TN | FP | FN | P | R | F1 | Improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generic | 389 | 221 | 741 | 25 | 0.34 | 0.93 | 0.50 | - |
| Domain-specific | 358 | 394 | 568 | 56 | 0.39 | 0.86 | 0.54 | 1.08x |

## LLM inference results

| Model | TP | TN | FP | FN | P | R | F1 | Improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Classifier (Baseline) | 358 | 568 | 0 | 0 | 0.38 | 0.5 | 0.43 | - |
| Llama-3.1-8B-Instruct | 332 | 441 | 127 | 26 | 0.72 | 0.92 | 0.81 | 1.86x |
| Llama-3-8B-Instruct | 297 | 361 | 207 | 61 | 0.59 | 0.83 | 0.69 | 1.6x |
| falcon-7b-instruct | 343 | 71 | 497 | 15 | 0.4 | 0.95 | 0.57 | 1.3x |
| Mistral-7B-Instruct-v0.3 | 32 | 512 | 56 | 326 | 0.36 | 0.089 | 0.14 | 0.33x |

## Manual inspection results

We calculate pairwise Cohen's kappa coefficient and then take an average of it.

#### Pair 1 cohen's kappa: 0.82

| | | Annotator 1 ||
| --- | --- | --- | --- |
| | | yes | no |
| **Annotator 2** | yes | 322 | 23 |
| | no | 27 | 228 |

#### Pair 2 cohen's kappa: 0.83

| | | Annotator 1 ||
| --- | --- | --- | --- |
| | | yes | no |
| **Annotator 3** | yes | 372 | 28 |
| | no | 17 | 183 |

#### Pair 3 cohen's kappa: 0.81

| | | Annotator 1 ||
| --- | --- | --- | --- |
| | | yes | no |
| **Annotator 4** | yes | 227 | 26 |
| | no | 16 | 185 |

#### **Average cohen's kappa: 0.82**