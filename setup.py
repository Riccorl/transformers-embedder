import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transformer_embedder",  # Replace with your own username
    version="1.0",
    author="Riccardo Orlando",
    author_email="orlandoricc@gmail.com",
    description="Word level transformer based embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Riccorl/transformer-embedder",
    keywords="NLP deep learning transformer pytorch BERT GPT GPT-2 google openai subtoken wordpieces",
    packages=setuptools.find_packages(),
    include_package_data=True,
    license="Apache",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch>=1.7,<1.9",
        "transformers>=4.3<4.4",
    ],
    python_requires=">=3.6",
)
