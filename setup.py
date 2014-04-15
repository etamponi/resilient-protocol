from distutils.core import setup

setup(
    name='resilient-protocol',
    version='0.2',
    packages=['resilient', 'resilient.configs'],
    url='https://github.com/etamponi/resilient-protocol',
    license='GPLv2',
    author='Emanuele Tamponi',
    author_email='emanuele.tamponi@gmail.com',
    description='Resilient Ensemble implementation using scikit-learn',
    requires=['ffnet', 'numpy', 'pybrain', 'matplotlib', 'scikit-learn'],
    scripts=['experiment_launcher']
)
