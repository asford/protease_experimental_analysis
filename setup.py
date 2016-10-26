from setuptools import setup, find_packages
import subprocess

setup(
    name="protease_experimental_analysis",
    packages=find_packages(),
    package_data={
        '': ['*.csv', '*.counts'],
    },
    install_requires=[
        l.strip() for l in open("requirements.txt").readlines()],
)
