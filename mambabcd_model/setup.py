from setuptools import setup, find_packages

setup(
    name="MambaCD",    # Changed from VMamba to MambaCD
    version="0.1",
    packages=find_packages(),
    package_dir={'': '.'},
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
