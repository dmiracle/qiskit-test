#!/bin/bash

python3 -m venv venv
. venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -Ur requirements.txt
. venv/bin/activate