import argparse
from importlib import import_module
import sys


COMMANDS = {
    "data": "tabbiomed.data.__main__",
    "run": "tabbiomed.trainer.__main__",
    "jobs": "tabbiomed.slurm",
    "results": "tabbiomed.results.__main__",
    "analyze": "tabbiomed.analysis.__main__",
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        command = sys.argv.pop(1)
        return import_module(COMMANDS[command]).main()
    parser = argparse.ArgumentParser(description="TabBioMed benchmark")
    parser.add_argument("command", choices=COMMANDS)
    parser.parse_args()


if __name__ == "__main__":
    main()
