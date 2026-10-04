import argparse
from caption_bench.calibration import analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    results = analyze(args.run)
    print(f"Evaluated {len(results)} non-identity caption/model pairs")
