from setuptools import setup, find_packages

setup(
    name='tfrecommender',
    version='0.1.1b1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='recommender systems using TensorFlow and embeddings',
    long_description=open('README.md').read(),
    install_requires=['numpy','pandas','tensorflow','matplotlib'],
    url='',
    author='Subarna Rana, Abhishek Kumar',
    author_email='subarna.rana2@gmail.com'
)