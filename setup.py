from setuptools import setup, find_packages

import generation

with open("README.md", 'r') as f:
    long_description = f.read()

# TODO: (@whiteRa2bit, 2020-08-25) Add pandas to requirements
# TODO: (@whiteRa2bit, 2020-08-25) Add tqdm to requirements
# TODO: (@whiteRa2bit, 2020-08-30) Add uproot to requirments
requirements = [
    "torch==1.4.0",
    "scipy==1.4.1",
    "numpy==1.18.1",
    "matplotlib==3.2.1",
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
