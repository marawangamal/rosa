#!/bin/bash

# Script Description:
# This script iterates through subdirectories within a given root directory, and for each subdirectory,
# evaluates reference and prediction text files using a provided executable file, appending the results to `metrics.txt`
# file.
#
# Usage:
# ./evaltools.sh <root_directory> <path_to_executable>
#
# Arguments:
# <root_directory>: The root directory containing the experiment folders.
# <path_to_executable>: The path to the executable file to run the evaluation.
#
# Details:
# - The script expects to find "e2e_test_references.txt" and "e2e_test_predictions.txt" files in each subdirectory.
# - The evaluation results are appended to a "metrics.txt" file in each respective subdirectory.
# - If the reference or prediction file is missing, an error message is recorded in the metrics file and printed


# Check if both the root directory and executable path are passed as arguments
if [ $# -ne 2 ]; then
  echo "Usage: $0 <path_to_executable> <root_directory>"
  exit 1
fi

# Define the root directory of the experiment folders from the first argument
executable="$1"
root="$2"

# Define the output file where the metrics will be saved
output_file_name="metrics.txt"

# Loop through all the experiment folders with the given root dir
for folder in "$root"/*; do
  if [ -d "$folder" ]; then
    echo "Processing folder: $folder"
    output_file="$folder/$output_file_name"

    # Define the paths for the reference and prediction txt files
    reference_file="$folder/test_references.txt"

    for file in "$folder"/*.csv; do
        mv "$file" "${file%.csv}.txt"
    done

    # iterate over prediction_files suffixed with _epoch_0, _epoch_1, etc.
    for prediction_file in "$folder"/test_predictions_latest*.txt; do
      # Check if both files exist before running the evaluation
      if [ -f "$reference_file" ] && [ -f "$prediction_file" ]; then
        echo "Evaluating $prediction_file" >> $output_file
        # Run the evaltools.py script and append the metrics to the output file
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
