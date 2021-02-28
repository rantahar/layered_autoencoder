from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["tensorflow", "numpy"]

setup(
    name='stackable_autoencoder',
    version='0.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=''
)
