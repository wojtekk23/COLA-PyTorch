import pandas as pd
import argparse
import sys

def filter_data(input_csv, excluded_labels, output_csv):
    """
    Read the data, filter out specific labels, and save the cleaned data.

    :param input_csv: str, path to the input CSV file (format similar to train.csv)
    :param excluded_labels: list of str, labels to exclude
    :param output_csv: str, path where the output CSV file will be saved
    """
    try:
        data = pd.read_csv(input_csv)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    def is_excluded(row):
        labels = row.split(',')
        return any(label in excluded_labels for label in labels)

    filtered_data = data[~data['positive_labels'].apply(is_excluded)]

    try:
        filtered_data.to_csv(output_csv, index=False)
        print(f"Filtered data saved to {output_csv}")
    except Exception as e:
        print(f"An error occurred while saving the CSV file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Filter rows from a CSV file based on excluded labels.")
    parser.add_argument(
            "input_csv",
            help="Path to the input CSV file (format similar to train.csv)"
            )
    parser.add_argument(
            "output_csv",
            help="Path where the output CSV file will be saved"
            )
    parser.add_argument(
            "--exclude",
            nargs='+',
            help="Labels to exclude. Provide each label separated by a space."
            )
    
    args = parser.parse_args()
    if args.exclude is None:
        print("No labels provided to exclude. Please provide labels using the '--exclude' option.", file=sys.stderr)
        sys.exit(1)

    filter_data(args.input_csv, args.exclude, args.output_csv)

if __name__ == "__main__":
    main()

