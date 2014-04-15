from distutils.core import setup

setup(
    name='resilient-protocol',
    version='0.2',
    packages=['configs', 'resilient'],
    url='',
    license='GPLv2',
    author='Emanuele Tamponi',
    author_email='emanuele.tamponi@gmail.com',
    description='Resilient Ensemble implementation using scikit-learn',
    requires=['ffnet', 'numpy', 'pybrain'],
    scripts=['experiment_launcher']
)
