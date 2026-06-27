"""Command-line interface — the single entrypoint for the Invoice AI system.

    invoice-ai <command> [options]

Commands: data, train, train-donut, tune, evaluate, extract, classify, demo-agent,
serve, benchmark, error-analysis, monitor, generate-report, generate-slides,
autopilot, grade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, ensure_dirs, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)


def _load(args) -> AppConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else AppConfig()
    ensure_dirs()
    return cfg


def cmd_data(args):
    from .data.download_dataset import download_all, download_task
    cfg = _load(args)
    res = download_all(cfg) if args.task == "all" else download_task(args.task, cfg)
    print(json.dumps(res, indent=2))


def cmd_train(args):
    from .training.train_layoutlmv3 import train_layoutlmv3
    print(json.dumps(train_layoutlmv3(_load(args), which=args.dataset, limit=args.limit), indent=2))


def cmd_train_donut(args):
    from .training.train_donut import train_donut
    print(json.dumps(train_donut(_load(args), limit=args.limit), indent=2))


def cmd_tune(args):
    from .training.tune import tune_layoutlmv3
    print(json.dumps(tune_layoutlmv3(_load(args), n_trials=args.n_trials), indent=2))


def cmd_evaluate(args):
    from .training.evaluate import evaluate
    print(json.dumps(evaluate(_load(args), which=args.dataset, limit=args.limit).get("summary", {}), indent=2))


def cmd_extract(args):
    from .agent.invoice_agent import InvoiceAgent
    agent = InvoiceAgent(_load(args))
    state = agent.process(doc_path=args.file, filename=Path(args.file).name)
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))


def cmd_classify(args):
    from .agent.invoice_agent import InvoiceAgent
    agent = InvoiceAgent(_load(args))
    sd = agent.process(doc_path=args.file, filename=Path(args.file).name).to_dict()
    print(json.dumps({"doc_type": sd["doc_type"], "confidence": sd["doc_type_conf"]}, indent=2))


def cmd_demo_agent(args):
    from .agent.invoice_agent import InvoiceAgent
    from .data.samples import SAMPLE_DOCS
    agent = InvoiceAgent(_load(args), load_model=False)
    for name, d in SAMPLE_DOCS.items():
        s = agent.process_tokens(d["tokens"], filename=name)
        v = s.validation
        print(f"\n{name}  ({d['note']})")
        print(f"  status     : {s.status.value}")
        print(f"  reconciles : {v.reconciles if v else '?'} (delta {v.reconcile_delta if v else '?'})")
        print(f"  fields     : {{'total': {s.fields.get('total').value if 'total' in s.fields else None}, "
              f"'subtotal': {s.fields.get('subtotal').value if 'subtotal' in s.fields else None}}}")
        if s.review_reasons:
            print(f"  review     : {s.review_reasons}")


def cmd_serve(args):
    import os
    import uvicorn
    if args.config:
        os.environ["INVOICE_AI_INFER_CONFIG"] = str(args.config)
    uvicorn.run("invoice_ai.api.main:app", host=args.host, port=args.port, reload=False)


def cmd_benchmark(args):
    from .analysis.latency import benchmark
    print(json.dumps(benchmark(_load(args), n=args.n, warmup=args.warmup), indent=2))


def cmd_error_analysis(args):
    from .analysis.error_analysis import error_analysis
    print(json.dumps(error_analysis(_load(args)), indent=2))


def cmd_monitor(args):
    from .monitoring.drift_report import monitoring_report
    print(json.dumps(monitoring_report(_load(args), log_path=args.log), indent=2))


def cmd_generate_report(args):
    from .autoreport.report_pdf import generate_report
    print("Report ->", generate_report(_load(args), title=args.title, author=args.author))


def cmd_generate_slides(args):
    from .autoreport.slides_pptx import generate_slides
    print("Slides ->", generate_slides(_load(args), title=args.title, author=args.author))


def cmd_autopilot(args):
    from .automation.autopilot import run_autopilot
    print(json.dumps(run_autopilot(_load(args), title=args.title, author=args.author,
                                   train=not args.no_train, limit=args.limit), indent=2))


def cmd_grade(args):
    from .grading.checklist import build_checklist
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[2]
    print(json.dumps(build_checklist(repo), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="invoice-ai", description="Invoice & Receipt Processing System")
    p.add_argument("--config", help="Path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("data", help="download datasets"); sp.add_argument("--task", choices=["all", "sroie", "cord", "funsd", "invoices"], default="all"); sp.set_defaults(func=cmd_data)
    sp = sub.add_parser("train", help="fine-tune LayoutLMv3 KIE"); sp.add_argument("--dataset", choices=["sroie", "funsd"], default="sroie"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_train)
    sp = sub.add_parser("train-donut", help="fine-tune Donut on CORD"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_train_donut)
    sp = sub.add_parser("tune", help="hyperparameter tuning"); sp.add_argument("--n-trials", type=int, default=5); sp.set_defaults(func=cmd_tune)
    sp = sub.add_parser("evaluate", help="entity-F1 + end-to-end agent eval"); sp.add_argument("--dataset", choices=["sroie", "funsd"], default="sroie"); sp.add_argument("--limit", type=int, default=50); sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("extract", help="extract one document"); sp.add_argument("--file", required=True); sp.set_defaults(func=cmd_extract)
    sp = sub.add_parser("classify", help="classify one document"); sp.add_argument("--file", required=True); sp.set_defaults(func=cmd_classify)
    sp = sub.add_parser("demo-agent", help="run the agent on built-in samples"); sp.set_defaults(func=cmd_demo_agent)
    sp = sub.add_parser("serve", help="start the FastAPI server"); sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8000); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("benchmark", help="latency benchmark"); sp.add_argument("--n", type=int, default=50); sp.add_argument("--warmup", type=int, default=5); sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("error-analysis", help="error analysis"); sp.set_defaults(func=cmd_error_analysis)
    sp = sub.add_parser("monitor", help="monitoring report from logs"); sp.add_argument("--log", default=None); sp.set_defaults(func=cmd_monitor)
    sp = sub.add_parser("generate-report", help="generate the PDF report"); sp.add_argument("--title", default="Invoice & Receipt Processing System"); sp.add_argument("--author", default="Le Dinh Minh Quan"); sp.set_defaults(func=cmd_generate_report)
    sp = sub.add_parser("generate-slides", help="generate the PPTX slides"); sp.add_argument("--title", default="Invoice & Receipt Processing System"); sp.add_argument("--author", default="Le Dinh Minh Quan"); sp.set_defaults(func=cmd_generate_slides)
    sp = sub.add_parser("autopilot", help="one-button: train -> eval -> analysis -> report+slides"); sp.add_argument("--title", default="Invoice & Receipt Processing System"); sp.add_argument("--author", default="Le Dinh Minh Quan"); sp.add_argument("--no-train", action="store_true"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_autopilot)
    sp = sub.add_parser("grade", help="rubric completeness self-check"); sp.add_argument("--repo", default=None); sp.set_defaults(func=cmd_grade)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
