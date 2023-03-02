from setuptools import setup
from pathlib import Path

with open(Path("requirements.txt"), "r") as requirements:
    dependencies = requirements.readlines()

setup(
    name='Data-Challenge-1-template',
    version='1.0.0',
    packages=['dc1'],
    url='',
    license='',
    author='',
    author_email='',
    description='',
    install_requires=dependencies,
)
