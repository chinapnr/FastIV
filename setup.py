import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FastIV",
    version="0.1",
    author="miaotianshi",
    author_email="miaotianshi@126.com",
    description=r'A package to compute weight of evidence (WOE) and '
                r'Information Value (IV) easily, with cross-features supported',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chinapnr/FastIV.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "scikit-learn>=0.20.3",
        "pandas>=0.25.1",
        "numpy>=1.16.2"
    ]
)
