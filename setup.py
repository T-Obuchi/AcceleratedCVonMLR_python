# coding=utf-8
from setuptools import setup, find_packages

setup(
    name='accelerated_cv_on_mlr',
    version='1.0.0a',
    description='alpha version of mean-field cross validation modules',
    url='https://github.com/T-Obuchi/AcceleratedCVonMLR_python',
    author='Takashi TAKAHASHI',
    author_email='takahashi.t.cc@m.titech.ac.jp',
    license='GPV-3',
    classifiers=[
        # How mature is this project
        'Development Status :: 3 - Alpha',

        # Indication of who this project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',

        # License
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specification of the Python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='lasso logistic regression mean-field statistical-physics',
    packages=find_packages(exclude=['samples']),
    install_requires=['numpy>1.13.0'],
    python_requires='>=3',
)
