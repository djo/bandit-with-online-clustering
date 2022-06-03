#!/bin/bash

black_code=0
flake_code=0
mypy_code=0
pytest_code=0

python -m black -S --line-length 100 ${justdoit:+--check} src
black_code=$?

python -m flake8 --max-line-length 100 src
flake_code=$?

python -m mypy src
mypy_code=$?

python -m pytest
pytest_code=$?

exit_code=0

for code in "black_code" "flake_code" "mypy_code" "pytest_code"; do
  if [ "${!code}" != 0 ]
  then
    echo "$code FAILED";
    exit_code=1;
  else
    echo "$code OK";
  fi
done

exit $exit_code;
