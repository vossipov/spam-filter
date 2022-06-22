from setuptools import setup, find_packages

setup(
    name='spam_filter',
    version='1.0',
    description='Spam filter implementation using Naive Bayes Classifier',
    author='Vyacheslav Osipov',
    url='https://github.com/vjaos/spam-filter',
    packages=find_packages(),
    install_requires=[
        'nltk==3.4.5',
        'numpy==1.22.0',
        'pandas==1.0.3',
        'python-dateutil==2.8.1',
        'pytz==2019.3',
        'six==1.14.0'
    ]
)
