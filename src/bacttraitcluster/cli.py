from __future__ import annotations
import argparse
import json

from .config import Config
from . import __version__
from .reliability import run_reliability_preflight, run_consistency_from_spec


def _build_parser():
    p = argparse.ArgumentParser(prog='bact-trait-cluster', description='Run bact-trait-cluster pipeline')
    p.add_argument('--config', default='examples/config.yaml', help='Path to YAML config')
    p.add_argument('--self-check', action='store_true', help='Run internal validation self-check and print JSON')
    p.add_argument('--self-check-report', default=None, help='Optional path to save self-check JSON')
    p.add_argument('--benchmark', action='store_true', help='Run synthetic benchmark and print JSON')
    p.add_argument('--benchmark-report', default=None, help='Optional path to save benchmark JSON')
    p.add_argument('--version', action='store_true', help='Print version and exit')
    p.add_argument('--reliability-preflight', action='store_true', help='Run reliability preflight before analysis')
    p.add_argument('--reliability-only', action='store_true', help='Run reliability preflight and exit (no analysis)')
    p.add_argument('--reliability-fail-fast', action='store_true', help='Abort when reliability quality_gate status is FAIL')
    p.add_argument('--reliability-sensitivity', action='store_true', help='Enable lightweight sensitivity mini-runs')
    p.add_argument('--reliability-config', default=None, help='YAML/JSON reliability thresholds preset')
    p.add_argument('--reliability-outdir', default=None, help='Output dir for reliability artifacts (default: <output_dir>/_reliability)')
    p.add_argument('--consistency-spec', default=None, help='YAML/JSON spec for cross-tool label consistency report')
    p.add_argument('--consistency-out', default=None, help='Output JSON path for consistency report')
    return p


def main(argv=None):
    p = _build_parser()
    args = p.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.self_check:
        from .selfcheck import run_self_check
        rep = run_self_check(args.self_check_report)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0 if rep.get('status') == 'PASS' else 1

    if args.benchmark:
        try:
            from .bench import synthetic_benchmark
        except Exception as e:
            rep = {"status": "SKIP", "reason": f"benchmark_unavailable: {e}"}
            print(json.dumps(rep, indent=2, ensure_ascii=False))
            return 0
        rep = synthetic_benchmark(report_path=args.benchmark_report)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0

    if args.consistency_spec:
        rep = run_consistency_from_spec(args.consistency_spec, args.consistency_out)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0 if rep.get('status') in ('PASS', 'WARN') else 1

    cfg = Config.from_yaml(args.config)

    # Persist config validation (unknown keys, schema version) into output_dir for auditability.
    try:
        from pathlib import Path
        outdir = Path(getattr(cfg, 'output_dir', 'results'))
        outdir.mkdir(parents=True, exist_ok=True)
        rep = getattr(cfg, '_config_validation', None)
        if isinstance(rep, dict):
            (outdir / 'config_validation.json').write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    except Exception:
        pass

    if args.reliability_preflight or args.reliability_only:
        rel = run_reliability_preflight(
            cfg,
            tool_name='bact-trait-cluster',
            tool_version=__version__,
            cfg_path=args.config,
            reliability_config_path=args.reliability_config,
            reliability_outdir=args.reliability_outdir,
            fail_fast=args.reliability_fail_fast,
            run_sensitivity=args.reliability_sensitivity,
        )
        print(json.dumps({'reliability_preflight': rel}, indent=2, ensure_ascii=False))
        if args.reliability_only:
            return 0 if rel.get('status') in ('PASS', 'WARN') else 1

    from .pipeline import Pipeline
    res = Pipeline(cfg).run()
    print(json.dumps({'status': 'ok', 'outputs': list(res.keys())}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
