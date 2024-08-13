#!/bin/bash

# Script to run exps

python_files = (
"exp/3_expsMinMax.py"
"exp/4_expsHLL.py"
"exp/9_expsPWL.py"
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