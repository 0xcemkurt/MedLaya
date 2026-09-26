# MedLaya

**Non-autoregressive System 1 decision engine for clinical triage.** Typed choice, score and yes/no decisions over patient messages and intake notes in a single forward pass — department routing, urgency scoring, and red-flag detection.

> Forked from [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), licensed under Apache 2.0. MedLaya adapts the original Laya decision engine for healthcare triage workflows.

---

## What it does

MedLaya takes any patient-facing text — an intake form, a portal message, a nurse's note, a symptom description in any of 100+ languages — and answers **typed questions** about it in one forward pass:

- **`choice`** — which department or care pathway should handle this (e.g. cardiology, general practice, emergency)
- **`score`** — how urgent the case is, on a defined ordinal scale
- **`noul`** — a calibrated probability for a yes/no flag (e.g. "does this describe a possible emergency symptom?")

Because it's a classification head over an encoder, not a text generator, there's nothing to hallucinate — the model can only ever return one of the answers you defined.

## Installation

```bash
python -m pip install -e .
```

Requires Python 3.10+. See [Installation details](https://github.com/NandhaKishorM/laya#installation-details) in the upstream repo for platform-specific notes (CPU/GPU PyTorch builds, Windows/macOS/Linux setup).

## Quickstart

```python
from medlaya import Router

router = Router(preload=True)

state = "I've had chest tightness and shortness of breath since this morning, it's getting worse."

questions = {
    "pathway": {
        "type": "choice",
        "instructions": "Which care pathway should handle this?",
        "criteria": {
            "emergency": "symptoms suggesting a medical emergency",
            "urgent_care": "needs same-day attention but not emergency",
            "primary_care": "routine, can be scheduled normally",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this case?",
        "criteria": ["routine", "same-day", "immediate"],
    },
    "red_flag": {
        "type": "noul",
        "instructions": "Does this describe a possible emergency symptom (e.g. cardiac, respiratory, neurological)?",
    },
}

result = router.predict(state, questions)
print(result["answers"]["pathway"]["choice"])     # e.g. emergency
print(result["answers"]["red_flag"]["noul"])      # probability, e.g. 0.94
print(result["routing"]["model"])                 # english / multilingual
```

The same call works across languages — the built-in router detects script/language and dispatches to the right checkpoint automatically.

## Checkpoints

| | encoder | context | use it for |
|---|---|---|---|
| `laya` (base) | ModernBERT-large | 512 | English |
| `laya-multilingual` | mmBERT-base | 1024 (up to 8192) | 100+ languages |
| `laya-typed-decisions` | ModernBERT-large | 1024 | general typed-decision workflows |

Fine-tuning on your own triage data (real intake logs, labelled by clinical staff) is where accuracy improves the most — see the upstream [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) as a starting point.

## Important: not a diagnostic tool

MedLaya classifies and routes text — it does **not** diagnose, and it is not a substitute for clinical judgment. Treat every output as a decision-support signal, not a final answer:

- Route low-confidence answers to a human reviewer rather than acting on them automatically.
- Validate accuracy and calibration on your own patient population before relying on any threshold — published benchmarks do not transfer directly to clinical data.
- Keep a human in the loop for anything touching emergency/red-flag detection.

## License

Apache 2.0 — see [`LICENSE`](./LICENSE). Original work © the Laya project (NandhaKishorM / Convai Innovations); modifications for clinical triage © [Your Company].
