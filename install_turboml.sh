#!/bin/bash

conda install -qq conda-forge::libstdcxx-ng anaconda::protobuf conda-forge::libtorch pytorch torchvision torchaudio cpuonly -c pytorch

pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple turboml

echo "Installation complete!"