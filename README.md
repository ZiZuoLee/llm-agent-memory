# LLM Agent Memory Mechanisms

Modern, focused evaluation of memory designs for LLM agents in controlled, multi-turn dialogue simulations. This project is research-oriented and compares memory strategies rather than deployment systems.

## At a Glance
- Research focus: comparative evaluation aligned with "Design and Implementation of Memory Mechanisms for Large Language Model Agents".
- Core entry points: `experiments/run_experiment.py`, `llm/llm_client.py`, `memory/*`, `prompt/prompt_builder.py`.
- Model setup: black-box LLM via API, no fine-tuning.

## Memory Modes
| Mode | Description |
| --- | --- |
| `no_memory` | No memory beyond the current turn. |
| `context` | Sliding window context memory. |
| `retrieval` | Vector-based retrieval memory. |
| `hierarchical` | Context + retrieval + structured profile memory. |

## Experimental Setup
- Multi-turn dialogue simulations.
- Evaluation signals: `preference_recall_correct` and `summary_mentions_preference`.
- Judgments require explicit, grounded attribution to earlier user statements.
- Scenarios include `short_preference` and `long_preference`.

## Run an Experiment
```bash
python experiments/run_experiment.py
```

## Results Summary
Long_preference scenario (repeat=5):

| Mode | Preference | Summary |
| --- | --- | --- |
| `no_memory` | 0.0 | 0.4 |
| `context` | 0.2 | 0.6 |
| `retrieval` | 1.0 | 0.8 |
| `hierarchical` | 1.0 | 1.0 |

## Notes and Limitations
- Results reflect controlled simulation settings and may not generalize to open-ended deployments.
- The LLM is treated as a black box; improvements are driven by memory design rather than model adaptation.
