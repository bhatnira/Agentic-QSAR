"""cta-qsar command-line interface."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cta-qsar",
        description="Chemically Trustworthy Agentic QSAR (CTA-QSAR)",
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", default=None, help="path to YAML config")
    shared.add_argument("--llm-provider", default=None, choices=["mock", "openrouter", "huggingface", "nvidia"])
    shared.add_argument("--llm-model", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", parents=[shared], help="profile a dataset (no experiments)")
    profile.add_argument("--data", required=True)
    profile.add_argument("--smiles-column")
    profile.add_argument("--target-column")
    profile.add_argument("--output", default=None, help="write profile JSON here")

    run = sub.add_parser("run", parents=[shared], help="run the full autonomous workflow")
    run.add_argument("--data", required=True)
    run.add_argument("--smiles-column")
    run.add_argument("--target-column")
    run.add_argument("--budget", type=int, default=None, help="max experiments")
    run.add_argument("--max-minutes", type=float, default=None)
    run.add_argument("--output", default=None, help="output directory")
    run.add_argument("--seed", type=int, default=None, help="global random seed")
    run.add_argument(
        "--hyperparameter-search",
        action="store_true",
        help="run budgeted grid search over model hyperparameters",
    )
    run.add_argument(
        "--no-hyperparameter-search",
        dest="hyperparameter_search",
        action="store_false",
        help="disable configured hyperparameter search",
    )

    report = sub.add_parser("report", help="re-render a report for a run id")
    report.add_argument("run_id")
    report.add_argument("--output", default=None)

    sub.add_parser("list-models", parents=[shared], help="list registered model plugins")
    sub.add_parser("list-representations", parents=[shared], help="list registered representation plugins")
    sub.add_parser("list-validation", parents=[shared], help="list registered validation plugins")
    sub.add_parser("list-providers", parents=[shared], help="list available LLM providers")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "profile":
            return _cmd_profile(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "list-models":
            return _cmd_list("model", "models")
        if args.command == "list-representations":
            return _cmd_list("representation", "representations")
        if args.command == "list-validation":
            return _cmd_list("validation", "validation plugins")
        if args.command == "list-providers":
            return _cmd_list_providers()
    except Exception as exc:  # noqa: BLE001
        print(f"cta-qsar error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    from cta_qsar.agents.scientist import QSARScientist
    from cta_qsar.core.config import build_config

    config = build_config(args.config, llm_provider=args.llm_provider, llm_model=args.llm_model)
    scientist = QSARScientist(config)
    state = scientist.profile(
        args.data, smiles_column=args.smiles_column, target_column=args.target_column
    )
    profile_payload = {
        "profile": state.get("profile", {}),
        "endpoint": state.get("endpoint", {}),
        "quality_report": state.get("quality_report", {}),
        "chemical_space": state.get("chemical_space", {}),
        "standardization": state.get("standardization_log", {}),
    }
    print(json.dumps(profile_payload, indent=2, default=str))
    if args.output:
        Path(args.output).write_text(json.dumps(profile_payload, indent=2, default=str))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from cta_qsar.agents.scientist import QSARScientist
    from cta_qsar.core.config import build_config
    from cta_qsar.core.logging import configure_logging

    configure_logging()
    config = build_config(
        args.config,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        max_experiments=args.budget,
        max_minutes=args.max_minutes,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        seed=args.seed,
        hyperparameter_search=args.hyperparameter_search,
    )
    scientist = QSARScientist(config, output_root=args.output)
    final = scientist.run(
        args.data,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        max_experiments=args.budget,
        max_minutes=args.max_minutes,
    )
    report = final if isinstance(final, dict) else {}
    print(f"Run complete. Report: {report.get('run_id', '')}")
    for exp in report.get("experiments", []):
        metrics = exp.get("metrics", {})
        if metrics:
            print(
                f"  {exp.get('representation')}+{exp.get('model')}[{exp.get('split')}]: {metrics}"
            )
    run_id = report.get("run_id")
    if run_id:
        output = Path(args.output or config.reporting.get("output_dir", "runs")) / run_id
        print(f"Artifacts: {output}/")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from cta_qsar.memory.experiment_memory import run_output_dir
    from cta_qsar.reporting.export import markdown_report

    run_dir = run_output_dir(args.output or "runs", args.run_id)
    report_path = run_dir / "report.json"
    if not report_path.exists():
        print(f"no report found at {report_path}", file=sys.stderr)
        return 1
    report = json.loads(report_path.read_text())
    print(markdown_report(report))
    return 0


def _cmd_list(kind: str, label: str) -> int:
    from cta_qsar.core.registry import get_registry

    registry = get_registry()
    with contextlib.suppress(Exception):
        registry.auto_discover()
    print(f"Registered {label}:")
    for name in registry.list(kind):
        print(f"  {name}")
    return 0


def _cmd_list_providers() -> int:
    from cta_qsar.llm.factory import list_providers

    providers = list_providers()
    print("Available LLM providers:")
    for spec in providers:
        env = f" (env: {', '.join(spec['requires_env'])})" if spec["requires_env"] else ""
        desc = f": {spec['description']}" if spec["description"] else ""
        print(f"  {spec['name']}{env}{desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())