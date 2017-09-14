pip install shapely
pip install scikit-learn
pip install pillow
pip install joblib
pip install cython
pip install opencv-python
pip install scikit-image
pip install pandas
pip install h5py
apt-get update
apt-get install libgeos-dev
apt-get install python-tk
apt-get install language-pack-en
cd Cytomine
#git clone https://github.com/cytomine/Cytomine-python-client.git
cd Cytomine-python-client/
cd client/
python setup.py build
python setup.py install
cd ..
cd utilities/
python setup.py build
python setup.py install
cd ..
cd ..
#git clone https://github.com/cytomine/Cytomine-python-datamining
cd Cytomine-python-datamining/cytomine-datamining/algorithms/pyxit/
cython _estimator.pyx
python setup.py build
python setup.py install
cd ~/../segmentation/TFE_Cytomine
curl -s get.sdkman.io | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk install groovy
git pull
