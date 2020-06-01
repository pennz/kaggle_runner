import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kaggle_runner",
    version="0.1.2",
    description="Run kaggle kernels, for fast model prototyping.",
    url="http://github.com/pennz/kaggle_runner",
    author="pennz",
    author_email="pengyuzhou.work@gmail.com",
    license="MIT",
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "slug",
        "parse",
        "python_logging_rabbitmq",
        "kaggle",
        "ipdb",  # later we can remove this
    ],
)
