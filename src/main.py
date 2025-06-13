import argparse
import subprocess
import os
import sys

def run_script(script_name, args):
    """Run a Python script with the given arguments"""
    cmd = [sys.executable, script_name] + args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="SERP Analysis Tool")
    parser.add_argument('input_file', type=str, help="Path to the input CSV file containing SERP data")
    parser.add_argument('--analysis', type=str, choices=['all', 'serp', 'domain', 'type'],
                        default='all', help="Type of analysis to run (default: all)")
    parser.add_argument('--keyword_column', type=str, default='keyword',
                        help="Name of the column containing keywords in the input CSV")
    parser.add_argument('--url_column', type=str, default='url',
                        help="Name of the column containing URLs in the input CSV")
    parser.add_argument('--min_common_urls', type=int, default=3,
                        help="Minimum number of common URLs required for keyword clustering")
    parser.add_argument('--min_overlap', type=float, default=0.1,
                        help="Minimum overlap coefficient required for keyword clustering")

    args = parser.parse_args()

    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)

    # Run selected analysis
    if args.analysis in ['all', 'serp']:
        print("\nRunning keyword clustering analysis...")
        clustering_args = [
            args.input_file,
            f"--keyword_column={args.keyword_column}",
            f"--url_column={args.url_column}",
            f"--min_common_urls={args.min_common_urls}",
            f"--min_overlap={args.min_overlap}",
            "--output_file=outputs/keyword_clusters.csv"
        ]
        run_script("src/serp_clustering.py", clustering_args)

    if args.analysis in ['all', 'domain']:
        print("\nRunning domain analysis...")
        domain_args = [args.input_file]
        run_script("src/domain_analysis.py", domain_args)

    if args.analysis in ['all', 'type']:
        print("\nRunning type distribution analysis...")
        type_args = [args.input_file]
        run_script("src/type_distribution.py", type_args)

    print("\nAnalysis completed successfully!")
    print("Results have been saved to the 'outputs' directory.")

if __name__ == "__main__":
    main()