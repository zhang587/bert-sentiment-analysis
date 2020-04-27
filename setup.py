import sys
from setuptools import setup, find_packages
import versioneer

setup(name='bert-sentiment-analysis',
      version=versioneer.get_version(),
      packages=find_packages(),
      install_requirements=[
          'transformers',
          'pandas==0.22',
          'pytorch==1.4.0',
          'torchvision==0.5.0'
      ]
)