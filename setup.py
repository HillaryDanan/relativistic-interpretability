from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="relativistic-interpretability",
    version="1.0.0",
    author="Hillary Danan",
    author_email="hillarydanan@gmail.com",
    description="A geometric framework for understanding neural network reasoning through multiple reference frames",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HillaryDanan/relativistic-interpretability",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyterlab>=3.2.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "plotly>=5.0.0",
            "networkx>=2.6.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "networkx>=2.6.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "relativistic-analyze=examples.minimal_example:main",
        ],
    },
)