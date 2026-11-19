import argparse
from pathlib import Path
from . import dataset_info, dataset_names, ensure_dataset, verify_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare the benchmark datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    for command in ("download", "verify", "prepare"):
        child = subparsers.add_parser(command)
        child.add_argument("datasets", nargs="*")
        child.add_argument("--data-root", type=Path, default=Path("data/processed"))
        if command == "prepare":
            child.add_argument(
                "--unified-root", type=Path, default=Path("data/unified_format")
            )
    from .curation import SOURCES

    child = subparsers.add_parser(
        "curate",
        help="Run reference raw-source curation independently of benchmark downloads",
    )
    child.add_argument("source", choices=SOURCES)
    child.add_argument("--workspace", type=Path, default=Path.cwd())
    child.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "list":
        for name in dataset_names():
            metadata = dataset_info(name)["metadata"]
            print(
                f"{name}\t{metadata['task']}\t{metadata['num_samples']}\t{metadata['num_features']}"
            )
        return
    if args.command == "curate":
        from .curation import curate

        arguments = (
            args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        )
        curate(args.source, arguments, workspace=args.workspace)
        return
    for name in args.datasets or dataset_names():
        if args.command == "download":
            ensure_dataset(name, args.data_root)
        elif args.command == "prepare":
            from .prepare import prepare_dataset

            prepare_dataset(name, args.unified_root, args.data_root)
        else:
            verify_dataset(name, args.data_root)
        print(name)


if __name__ == "__main__":
    main()
