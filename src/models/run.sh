#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the linear regression script
python linear_regression.py

# Run the decision tree script
python decision_tree.py

echo "Both scripts ran successfully."

python upload.py