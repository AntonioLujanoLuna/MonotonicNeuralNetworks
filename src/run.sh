#!/bin/bash

# Activate the virtual environment
source /home/antoniolujano/.virtualenvs/MonotonicNeuralNetworks/bin/activate

# Add the parent directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Script to run exps

python_files=(
#"exp/9_expsPWL.py"
"exp/10_expsCMNN.py"
"exp/11_expsPWLMixup.py"
)

for file in "${python_files[@]}"; do
  echo "Running $file"
  python3 "$file"
  if [ $? -ne 0 ]; then
      echo "Error while running $file"
      exit 1
  fi
done

echo "All executed"

# Deactivate the virtual environment when done
deactivate