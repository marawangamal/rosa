#!/bin/bash

# Check if the root directory is passed as an argument
if [ $# -eq 0 ]; then
  echo "Please provide the root directory as an argument."
  exit 1
fi

# Define the root directory of the experiment folders from the first argument
root="$1"

executable=/home/mila/a/aristides.milios/scratch/gpt2-e2e/e2e-metrics/measure_scores.py

# Define the output file where the metrics will be saved
output_file_name="metrics.txt"

# Loop through all the experiment folders with the given root dir
for folder in "$root"/*; do
  if [ -d "$folder" ]; then
    echo "Processing folder: $folder"
    output_file="$folder/$output_file_name"

    # Define the paths for the reference and prediction txt files
    reference_file="$folder/e2e_test_references.txt"

    # iterate over prediction_files suffixed with _epoch_0, _epoch_1, etc.
    for prediction_file in "$folder"/e2e_test_predictions_epoch_*.txt; do
      # Check if both files exist before running the evaluation
      if [ -f "$reference_file" ] && [ -f "$prediction_file" ]; then
        echo "Evaluating $prediction_file" >> $output_file
        # Run the eval.py script and append the metrics to the output file
        $executable -p "$reference_file" "$prediction_file" >> $output_file
        echo "" >> $output_file # Add an empty line to separate results
      else
        echo "Missing reference or prediction file in $folder" # This will print to the console
        echo "Missing reference or prediction file in $folder" >> $output_file
      fi
    done
  else
    echo "$folder is not a directory, skipping..."
  fi
done

echo "Evaluation complete. Results are saved in $output_file"
