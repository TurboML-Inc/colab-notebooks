#!/bin/bash

conda install -qq --no-pin python=3.11 conda-forge::libstdcxx-ng anaconda::protobuf conda-forge::libtorch conda-forge::pytorch conda-forge::torchvision conda-forge::torchaudio conda-forge::ncurses

cp -r /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py /usr/local/lib/python3.10/dist-packages/google-2.0.3.dist-info /usr/local/lib/python3.10/dist-packages/google_colab-1.0.0.dist-info /usr/local/lib/python3.11/site-packages
cp -r /usr/local/lib/python3.10/dist-packages/google/* /usr/local/lib/python3.11/site-packages/google
sed -i "s/from IPython.utils import traitlets as _traitlets/import traitlets as _traitlets/" /usr/local/lib/python3.11/site-packages/google/colab/*.py
sed -i "s/from IPython.utils import traitlets/import traitlets/" /usr/local/lib/python3.11/site-packages/google/colab/*.py
python -m pip install ipython traitlets jupyter psutil matplotlib setuptools ipython_genutils ipykernel jupyter_console prompt_toolkit httplib2 astor google-auth==2.27.0 ipyparallel==8.8.0 pandas==2.2.2 portpicker==1.5.2 ipykernel==5.5.6 ipython==7.34.0 notebook==6.5.5 requests==2.32.3 tornado==6.3.3
sed -i 's|/usr/bin/python3\.real|/usr/local/bin/python|g' /usr/bin/python3
python -m pip install turboml-sdk

echo "Installation complete!"
