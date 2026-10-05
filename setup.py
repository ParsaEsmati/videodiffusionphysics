from pathlib import Path

from setuptools import setup

root = Path(__file__).parent

setup(
    name="video-phys",
    version="0.1.0",
    description="Experimental tooling for sampling and probing experiments on video diffusion model features.",
    long_description=(root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="",
    python_requires=">=3.10",
    install_requires=[
        "diffusers",
        "transformers",
        "torch",
        "numpy",
        "huggingface_hub",
    ],
    py_modules=["main"],
    entry_points={
        "console_scripts": [
            "flow-guidance=main:main",
        ]
    },
)
