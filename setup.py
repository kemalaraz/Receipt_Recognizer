from setuptools import setup, find_packages

# load long description
with open("README.md", "r") as foo:
    long_description = foo.read()

# load requirements
with open("requirements.txt", "r") as foo:
    requirements = foo.read().split("\n")

setup(
    name="receiptrecognizer",
    version="0.0.1",
    packages=find_packages(),
)

setup(
    # package name `pip install fastface`
    name="receiptrecognizer",
    # small description
    description="",
    # long description
    long_description=long_description,
    # content type of long description
    long_description_content_type="text/markdown",
    # source code url for this package
    url="https://github.com/kemalaraz/Receipt_Recognizer",
    # package license
    license='MIT',
    # package root directory
    packages=find_packages(),
    # requirements
    install_requires=requirements
)