import argparse
from pathlib import Path
import subprocess
from .paths import GENOMIC_ANNOTATIONS_DIR


def prepare_subsets(raw_root, *, executable="plink2", threads=10):
    root = Path(raw_root).expanduser().resolve()

    def run(*arguments):
        subprocess.run(
            [
                executable,
                *map(str, arguments),
                "--threads",
                str(threads),
                "--seed",
                "42",
            ],
            cwd=root,
            check=True,
        )

    run(
        "--vcf",
        "gnomad.genomes.v3.1.2.hgdp_tgp.chrX.vcf.bgz",
        "--make-pgen",
        "--update-sex",
        GENOMIC_ANNOTATIONS_DIR / "sexcheck_labels.txt",
        "--split-par",
        "hg38",
        "--out",
        "chrX_with_sex",
    )
    run(
        "--vcf",
        "gnomad.genomes.v3.1.2.hgdp_tgp.chr22.vcf.bgz",
        "--make-pgen",
        "--out",
        "chr22_full",
    )
    for prefix, widths in (
        ("chr22_full", (50000, 1000)),
        ("chrX_with_sex", (50000, 500, 50)),
    ):
        chromosome = "chr22" if prefix == "chr22_full" else "chrX"
        for width in widths:
            suffix = "50k" if width == 50000 else "1k" if width == 1000 else str(width)
            run(
                "--pfile",
                prefix,
                "--thin-count",
                width,
                "--make-pgen",
                "--out",
                f"{chromosome}_{suffix}",
            )
    ids = []
    with (root / "chr22_full.pvar").open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if 750000 <= line_number <= 750999:
                fields = line.split()
                ids.append(fields[0] + ":" + fields[1])
            if line_number == 750999:
                break
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError(
            "The chromosome 22 source does not contain the expected consecutive block"
        )
    (root / "ids_1000.txt").write_text("\n".join(ids) + "\n")
    run(
        "--pfile",
        "chr22_full",
        "--set-all-var-ids",
        "@:#",
        "--extract",
        "ids_1000.txt",
        "--make-pgen",
        "--out",
        "chr22_1000_consec",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare the selected genomic source subsets"
    )
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--plink2", default="plink2")
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    prepare_subsets(args.raw_root, executable=args.plink2, threads=args.threads)


if __name__ == "__main__":
    main()
