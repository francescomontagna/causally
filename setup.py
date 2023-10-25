"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.


# See https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#setup-py
# for arguments reference
# See https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/ for a quick, useful, guide

setup(
    name="causally",  # Required
    version="0.1.0",  # Required
    description="Semi-sytnthetic generation of causl discovery data in realistic scenarios",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",
    url="https://github.com/francescomontagna/causally",
    author="Francesco Montagna",  # Optional
    author_email="francesco.montagna997@gmail.com",  # Optional
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src", exclude="*.py"),  # Required
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.25.2",
        "networkx>=3.1",
        "scikit-learn>=1.3.0",
        "torch>=2.0.1"
    ], # Optional
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest']
)