from setuptools import setup, find_packages

setup(
    name='sumformer2',
    version='0.3',
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open('requirements.txt', 'r')
    ],
    entry_points={
        'console_scripts': [
            'sumformer2=sumformer2.main:start',
        ],
    },
    author='Ryan Ott',
    author_email='ott.r21@gmail.com',
    description='Summarisation Transformer 2',
    url='https://github.com/Ryan-Ott/sumformer2',
)