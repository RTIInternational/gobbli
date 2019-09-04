import json
import os

from setuptools import find_packages, setup

containing_dir = os.path.split(__file__)[0]

with open(os.path.join(containing_dir, "meta.json"), "r") as f:
    meta = json.load(f)

setup(
    name=meta["name"],
    author=meta["author"],
    maintainer=meta["maintainer"],
    version=meta["version"],
    packages=find_packages(exclude=["test"]),
    license="LICENSE.txt",
    description=meta["description"],
    install_requires=[
        "docker >= 3.7.0",
        "numpy >= 1.16.0",
        "pandas >= 0.24.0",
        "scikit-learn >= 0.21.0",
        "ray >= 0.7.3",
        "seaborn >= 0.9.0",
    ],
    extras_require={
        "augment": ["nltk>=3.4.4", "gensim>=3.8.0", "spacy>=2.1.4"],
        "tokenize": ["sentencepiece >= v0.1.83"],
    },
    python_requires=">=3.7",
    entry_points={"console_scripts": []},
)
