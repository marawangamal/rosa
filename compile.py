import os
import csv

# Define the root directory
root = '.'

# Define the output CSV file
output_csv = 'results.csv'

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header row to the CSV file
    writer.writerow(['Subdirectory', 'BLEU', 'NIST', 'METEOR', 'ROUGE_L', 'CIDEr'])

    # Loop through the subdirectories in the root directory
    for subdir, _, _ in os.walk(root):
        # Check if the subdirectory contains a file named 'metrics.txt'
        metrics_file_path = os.path.join(subdir, 'metrics.txt')
        if os.path.isfile(metrics_file_path):
            print(f'Processing folder: {subdir}')

            # Read the metrics from the file
            with open(metrics_file_path, 'r') as metrics_file:
                lines = metrics_file.readlines()
                scores = {line.split(":")[0].strip(): line.split(":")[1].strip() for line in lines if ':' in line}

            # Write the metrics to the CSV file
            writer.writerow([
                subdir,
                scores.get('BLEU', ''),
                scores.get('NIST', ''),
                scores.get('METEOR', ''),
                scores.get('ROUGE_L', ''),
                scores.get('CIDEr', '')
            ])

print(f'Evaluation complete. Results are saved in {output_csv}')
