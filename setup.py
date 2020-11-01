from setuptools import setup, find_packages

import generation

with open("README.md", 'r') as f:
    long_description = f.read()

# TODO: (@whiteRa2bit, 2020-09-23) Add skimage
requirements = [
    "torch==1.4.0",
    "scipy==1.4.1",
    "numpy==1.18.1",
    "matplotlib==3.2.1",
    "wandb==0.10.5",
    "uproot==3.12.0",
    "tqdm==4.48.2",
    "loguru==0.5.3",
    "pandas==1.1.0",
    "plotly==4.10.0"
]

setup(
    name='generation',
    version=generation.__version__,
    description='Module of generative models applied to high energy physics problems',
    license="MIT",
    long_description=long_description,
    author='Pavel Fakanov',
    author_email='pavel.fakanov@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
)
