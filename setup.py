import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

extras = {}
extras["torch"] = ["torch>=1.5,<1.11"]
extras["spacy"] = ["spacy>=3.0,<3.3"]
extras["all"] = extras["torch"] + extras["spacy"]

install_requires = ["transformers>=4.3,<4.17"]

setuptools.setup(
    name="transformers_embedder",  # Replace with your own username
    version="1.8.5",
    author="Riccardo Orlando",
    author_email="orlandoricc@gmail.com",
    description="Word level transformer based embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Riccorl/transformers-embedder",
    keywords="NLP deep learning transformer pytorch BERT google subtoken wordpieces embeddings",
    packages=setuptools.find_packages(),
    include_package_data=True,
    license="Apache",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    extras_require=extras,
    install_requires=install_requires,
    python_requires=">=3.6",
)
