#!/bin/bash

conda install -qq conda-forge::libstdcxx-ng anaconda::protobuf conda-forge::libtorch conda-forge::pytorch conda-forge::torchvision conda-forge::torchaudio

pip install turboml-sdk

echo "Installation complete!"
