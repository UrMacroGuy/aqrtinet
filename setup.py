from setuptools import setup, find_packages

setup(
    name="aqrtinet",
    version="0.1.0",
    description="Custom regime-aware gradient booster for NSE/BSE stock direction prediction",
    author="Pratyush Dave",
    author_email="pratyushdave80@gmail.com",
    url="https://github.com/pratyushdave80/aqrtinet",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "examples": ["yfinance>=0.2.0"],
        "test":     ["pytest>=7.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Office/Business :: Financial :: Investment",
        "Intended Audience :: Financial and Insurance Industry",
    ],
)
