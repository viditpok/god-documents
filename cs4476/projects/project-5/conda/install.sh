#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ENVIRONMENT_FILE=requirements.txt
echo "Using ${ENVIRONMENT_FILE} ..."

# Create library environment.
# conda env create -f "${SCRIPT_DIR}/${ENVIRONMENT_FILE}" \
# && eval "$(conda shell.bash hook)" \
# && conda activate cv_proj1 \
# && python -m pip install -e "$SCRIPT_DIR/.."

conda create -n cv_proj5 python=3.10 \
&& conda activate cv_proj5 \
&& pip3 install -r ./requirements.txt
&& pip install -e .