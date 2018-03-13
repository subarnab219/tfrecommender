from setuptools import setup, find_packages

setup(
    name='tfrecommender',
    version='0.1.1dev3',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='recommender systems using TensorFlow and embeddings',
    long_description=open('README.md').read(),
    install_requires=['numpy','pandas','tensorflow','matplotlib'],
    url='https://github.com/subarnab219/tfrecommender.git',
    author='Subarna Rana, Abhishek Kumar',
    author_email='subarna.rana2@gmail.com'
)