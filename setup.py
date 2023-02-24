import os
from setuptools import setup, find_packages

path = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(path, 'README.md'), "r", encoding='utf-8') as f:
        long_description = f.read()
except Exception as e:
    long_description = "scAce: an adaptive embedding and clustering method for single-cell gene expression data"

setup(
    name="scace",
    version="0.1.0",
    keywords=["single-cell RNA-sequencing", "clustering", "cluster merging"],
    description="scAce: an adaptive embedding and clustering method for single-cell gene expression data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT Licence",

    url="https://github.com/sldyns/scAce",
    author="Kun Qian, Xinwei He",
    author_email="kun_qian@foxmail.com",
    maintainer="Kun Qian",
    maintainer_email="kun_qian@foxmail.com",

    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scanpy",
        "torch",
        "pandas",
        "tqdm",
        "scipy",
        "sklearn"
        ],
    platforms='any'
)
