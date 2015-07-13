# Ref: http://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2
from setuptools import setup


setup(name='ProbabPy',
      version='0.1.3',
      packages=['ProbabPy'],
      url='https://github.com/MBALearnsToCode/ProbabPy',
      author='Vinh Luong (a.k.a. MBALearnsToCode)',
      author_email='MBALearnsToCode@UChicago.edu',
      description='(Multivariate) Probability Distributions',
      long_description='(please read README.md on GitHub repo)',
      license='MIT License',
      install_requires=['CompyledFunc', 'FrozenDict', 'HelpyFuncs', 'MathDict', 'MathFunc', 'SciPy', 'SymPy'],
      classifiers=[],   # https://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='prob probability distribution')
