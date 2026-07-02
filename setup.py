from setuptools import setup, find_packages

setup(
    name="pcb-router",
    version="0.1.0",
    description="AI-powered PCB routing with GNN + JEPA architecture",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
