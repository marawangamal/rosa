import os
import pandas as pd

# Define the root directory
root = '.'

# Define the output CSV file
output_csv = 'results.csv'


def rename_function(filename):
    peft_name = filename.split('_name')[-1].split('_')[0]
    model_name = filename.split('_name')[0].split('_nam')[-1].split('_')[0]
    rank = filename.split('_r')[-1].split('_')[0]
    new_name = f'{model_name}_{peft_name}_{rank}'
    return new_name

# Create a DataFrame to hold the results
df = pd.DataFrame(columns=['Subdirectory', 'BLEU', 'NIST', 'METEOR', 'ROUGE_L', 'CIDEr'])

# Create a list to hold the results
results = []

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

        # Append the metrics to the results list
        results.append([
            rename_function(subdir),
            scores.get('BLEU', ''),
            scores.get('NIST', ''),
            scores.get('METEOR', ''),
            scores.get('ROUGE_L', ''),
            scores.get('CIDEr', '')
        ])

# Convert the results list to a DataFrame
df = pd.DataFrame(results, columns=['Subdirectory', 'BLEU', 'NIST', 'METEOR', 'ROUGE_L', 'CIDEr'])
df = df.sort_values('Subdirectory')

# Print the DataFrame
print(df)
latex_table = df.to_latex(index=False)
print(latex_table)

# Save the DataFrame to CSV
df.to_csv(output_csv, index=False)

print(f'Evaluation complete. Results are saved in {output_csv}')
