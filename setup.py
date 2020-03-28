from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='generation',
   version='0.1',
   description='Module of generative models applied to high energy physics problems',
   license="MIT",
   long_description=long_description,
   author='Pavel Fakanov',
   author_email='pavel.fakanov@gmail.com',
   packages=find_packages(),
)
