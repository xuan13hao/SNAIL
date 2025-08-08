# snail_xgb/cli.py
import os
import sys
import argparse
from importlib.resources import files


def _data_dir() -> str:
    # works even if data/ isn’t a package
    return str(files("snail_xgb").joinpath("data"))

def main():
    parser = argparse.ArgumentParser(prog="snail-xgb")
    parser.add_argument("input_file", help="Path to input TXT (e.g., PCB_articles.txt)")
    parser.add_argument("output_csv", help="Path to output CSV (e.g., test.csv)")
    args = parser.parse_args()

    # Make user-provided paths absolute BEFORE changing cwd
    input_abs = os.path.abspath(args.input_file)
    output_abs = os.path.abspath(args.output_csv)

    # Change cwd to the package data folder so relative resources in SNAIL.py resolve:
    # dbsoftDB.tsv, acronyms.dic, bioconductor_packages.txt, scowl/, best_xgb_model.joblib
    os.chdir(_data_dir())

    # Rebuild argv for SNAIL.main() which expects: script.py <input_file> <output_csv>
    sys.argv = ["SNAIL.py", input_abs, output_abs]

    # Import and run your existing script's main()
    from .SNAIL import main as snail_main
    snail_main()

if __name__ == "__main__":
    main()
