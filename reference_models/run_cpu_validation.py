"""Run the actual synthetic CPU tests and save their honest outcome.

No external data, pretrained weights, benchmark metrics, GPU or network calls.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Validation output must be new")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parent
    files = sorted((root / "genref").glob("*.py")) + sorted((root / "tests").glob("*.py"))
    files += [Path(__file__).resolve()]
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    started = time.monotonic()
    report = {"status": "NOT_RUN", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Synthetic CPU software/mathematics checks; not model efficacy or a dataset benchmark",
              "gpu_calls": False, "external_dataset_or_network_calls": False,
              "python": platform.python_version(), "source_sha256": before}
    try:
        import torch
        torch.set_num_threads(1)
        report["torch"] = torch.__version__
        report["device"] = "cpu"
        from tests import test_reference_models as tests
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(tests))
        (args.output / "unittest.log").write_text(stream.getvalue())
        report.update(status="PASSED" if result.wasSuccessful() else "FAILED",
                      tests_run=result.testsRun, failures=len(result.failures),
                      errors=len(result.errors), skipped=len(result.skipped),
                      model_checks=tests.RESULTS)
        if not result.wasSuccessful():
            print(stream.getvalue(), file=sys.stderr)
    except BaseException as error:
        report.update(status="UNAVAILABLE" if isinstance(error, ModuleNotFoundError) else "FAILED",
                      error_type=type(error).__name__, error=str(error))
    after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report["source_unchanged"] = before == after
    report["elapsed_seconds"] = time.monotonic() - started
    if before != after:
        report["status"] = "FAILED"
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ["source_sha256", "model_checks"]}))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

