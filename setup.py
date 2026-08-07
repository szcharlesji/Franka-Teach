from setuptools import find_packages, setup

setup(
    name="frankateach",
    version="0.0.1",
    packages=find_packages(),
    package_data={
        "frankateach.airhockey": ["static/*.html"],
        "frankateach.recording": ["static/*.html"],
    },
    install_requires=["gymnasium"],
)
