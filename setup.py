from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

requirements = [
    "torch==1.4.0",
    "scipy==1.4.1",
    "numpy==1.18.1",
    "matplotlib==3.2.1",
]

setup(
    name='generation',
    version='0.1.2',
    description='Module of generative models applied to high energy physics problems',
    license="MIT",
    long_description=long_description,
    author='Pavel Fakanov',
    author_email='pavel.fakanov@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
)
