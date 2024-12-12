#!/bin/bash

conda install -qq conda-forge::libstdcxx-ng anaconda::protobuf

gdown --quiet "1UG8cmBLdYtEVJZG8Rgz454l722VAt_zE"

unzip -qq turboml.zip -d turboml

pip install -qq turboml/*.whl

rm -rf turboml turboml.zip

echo "Installation complete!"