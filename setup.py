"""
Setup script for Multi-Speaker ASR with NLP Enhancement
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multi-speaker-asr-nlp",
    version="0.1.0",
    author="dhiaa-a",
    author_email="dhiaajhamed@gmail.com",
    description="NLP-Enhanced Error Correction for Multi-Speaker ASR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhiaa-a/multi-speaker-asr-nlp",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "torch>=1.9.0+cu111",
            "torchaudio>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "asr-demo=scripts.run_demo:main",
            "asr-server=scripts.run_server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)