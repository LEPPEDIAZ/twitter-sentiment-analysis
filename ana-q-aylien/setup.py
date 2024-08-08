from setuptools import setup

with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="ana_q_aylien",
    version=version,
    packages=["ana_q_aylien"],
)
