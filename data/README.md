# `data/` — datasets & download scripts

**Large datasets / document images are never committed** (see `.gitignore`).

```bash
python -m invoice_ai.cli data --task all      # SROIE + CORD + FUNSD (HF cache)
python -m invoice_ai.cli data --task sroie    # one dataset
```

Verified dataset ids, schemas and licenses are in
[`../docs/data_description.md`](../docs/data_description.md). LayoutLMv3 training
uses `mp-02/sroie` (image + words + boxes + NER tags); Donut uses
`naver-clova-ix/cord-v2` (image + ground-truth JSON).
