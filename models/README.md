# `models/` — trained models & checkpoints

**Nothing here is committed to Git** (see `.gitignore`). Training writes here (or
to a Drive/`ARTIFACTS_DIR` on Colab); inference loads from `INVOICE_AI_MODEL_DIR`.

If no fine-tuned model is present, the agent uses the **regex baseline extractor**
so the system always runs offline.

```
models/
├── layout_extractor/latest/   # fine-tuned LayoutLMv3 KIE model + labels.json + model_metadata.json
├── donut/latest/              # (optional) Donut image->JSON parser
└── README.md
```
