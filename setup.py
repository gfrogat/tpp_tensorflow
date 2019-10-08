import subprocess
from pathlib import Path

from setuptools import find_packages, setup

cwd = Path("").absolute()

version = "0.2.0"
sha = "Unknown"

version_path = cwd / "tpp_tensorflow" / "version.py"

with open(version_path, "w") as f:
    f.write(f'__version__ = "{version}"\n')
    f.write(f'git_version = "{sha}"')


setup(
    # Metadata
    name="tpp_tensorflow",
    version=version,
    author="Peter Ruch",
    author_email="tpp@ml.jku.at",
    license="MIT",
    description="Target Prediction Platform TensorFlow Code",
    # Package info
    packages=find_packages(),
    zip_safe=False,
)
