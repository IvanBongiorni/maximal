"""
Maximal setup

MIT License

Copyright (c) 2022 Ivan Bongiorni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from setuptools import setup, find_packages

classifiers=[
        'Development Status :: 4 - Beta',   # 3: 'Alpha'; 4: 'Beta'; 5: 'Production/Stable'
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: AI :: Machine Learning :: NLP :: Natural Language Processing'
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
  ]

setup(
    name='maximal',
    version=0.3,
    description='TensorFlow-compatible Transformer layers and models.',
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    url='https://github.com/IvanBongiorni/maximal',
    author='Ivan Bongiorni',
    author_email='ivanbongiorni@protonmail.com'
    license='MIT',
    classifiers=classifiers,
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow>=2.0'
    ]
)
