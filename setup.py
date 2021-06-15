from setuptools import setup, find_packages

# load long description
with open("README.md", "r") as foo:
    long_description = foo.read()

# load requirements
with open("requirements.txt") as foo:
    required = foo.read().splitlines()

setup(
    name="receiptrecognizer",
    version="0.0.1",
    packages=find_packages(),
)

setup(
    # package name `pip install fastface`
    name="receiptrecognizer",
    version="0.0.1",
    author="karaz",
    author_email="kemalaraz91@gmail.com",
    # small description
    description="Receipt Recognition Package",
    # long description
    long_description=long_description,
    # content type of long description
    long_description_content_type="text/markdown",
    # source code url for this package
    url="https://github.com/kemalaraz/Receipt_Recognizer",
    # package license
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=required,
)