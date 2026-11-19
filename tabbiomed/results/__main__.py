"""Command-line interface for benchmark result consolidation."""

import argparse
import importlib
import json
import sys

from .consolidation import consolidate


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and consolidate explicit benchmark runs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "consolidate", help="Prepare reports, predictions, labels and timings."
    )
    command.add_argument(
        "--manifest", required=True, help="Versioned JSON with explicit report paths."
    )
    command.add_argument(
        "--output", required=True, help="Public result collection directory."
    )
    command.add_argument(
        "--log-dir",
        help="Optionally save consolidation logs in this directory.",
    )
    command.add_argument(
        "--execute",
        action="store_true",
        help="Install validated evaluations. Omit to preflight.",
    )
    command.add_argument("--model-module")
    manifest = commands.add_parser(
        "manifest", help="Collect unambiguous settled raw runs"
    )
    manifest.add_argument("--raw", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--model-module")
    args = parser.parse_args(argv)
    if args.model_module:
        importlib.import_module(args.model_module)
    if args.command == "manifest":
        from .manifest import create_manifest

        print(create_manifest(args.raw, args.output))
        return 0
    try:
        receipt = consolidate(
            args.manifest, args.output, args.log_dir, execute=args.execute
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Consolidation failed: {error}\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "evaluations": len(receipt["evaluations"]),
                "output": receipt["output"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
