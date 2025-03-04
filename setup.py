from setuptools import setup, find_packages

setup(
    name='EGAT',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'pandas',  # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here if needed
        ],
    },
    include_package_data=True,
    description='A description of your package',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/EGAT',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)