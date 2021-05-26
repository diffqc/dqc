echo "Installing python requirements"
python -m pip install --upgrade pip
python -m pip install -r test-requirements.txt
python -m pip install -e .

echo "Installing libraries from PySCF"
cd lib; mkdir build; cd build
cmake ..
make
cd ../..

echo "Installing libcint"
cd lib/libcint; mkdir build; cd build
cmake ..
make
cd ../../..

echo "Installing libxc"
cd submodules/libxc
python setup.py install
cd ../..
