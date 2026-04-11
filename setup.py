from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brain-tumour-segmentation",
    version="1.0.0",
    author="Karthik Pagnis",
    author_email="karthikpagnis@iitm.ac.in",
    description="Brain Tumour Segmentation from MRI Scans Using Attention-Enhanced U-Net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karthikpagnis/Brain-Tumour-Segmentation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    keywords=[
        "healthcare",
        "deep-learning",
        "segmentation",
        "medical-imaging",
        "mri",
        "brain-tumor",
        "u-net",
        "attention-mechanisms",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/karthikpagnis/Brain-Tumour-Segmentation/issues",
        "Source": "https://github.com/karthikpagnis/Brain-Tumour-Segmentation",
    },
)
