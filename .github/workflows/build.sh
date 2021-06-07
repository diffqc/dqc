echo "Installing python requirements"
python -m pip install --upgrade pip
python -m pip install -r test-requirements.txt
python -m pip install -e .
