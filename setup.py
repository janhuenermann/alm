import setuptools

setuptools.setup(
    name="alm",
    version="0.0.1",
    author="Jan Hünermann",
    author_email="mail@janhuenermann.com",
    description="Algorithms for ML",
    long_description_content_type="text/markdown",
    url="https://github.com/janhuenermann/alm",
    project_urls={
        "Bug Tracker": "https://github.com/janhuenermann/alm/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    package_data={'': ['geometry/native/**/*']},
    include_package_data=True,
)
