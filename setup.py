# Copyright (c) 2026 MoganAI.
# Derived from FlashQLA (Copyright (c) 2026 The Qwen team, Alibaba Group). Licensed under The MIT License [see LICENSE].

from setuptools import setup, find_packages

setup(
    name="mogan-kda",
    version="0.1.0",
    description="MoganKDA: Hopper TileLang kernels for Kimi Delta Attention (KDA) training and inference",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/moganai/MoganKDA",
    packages=find_packages(include=["mogan_kda", "mogan_kda.*"]),
    package_data={"": ["coefs/*.csv"]},
    license="MIT",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.8",
        "tilelang==0.1.12",
    ],
    extras_require={"test": ["pytest", "flash-linear-attention"]},
    zip_safe=False,
)
