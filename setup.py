from setuptools import find_packages, setup

setup(
    name='peftnet',
    packages=find_packages(),
    # package_dir={'cprnn': 'src'},
    version='0.1.0',
    description='Parameter Efficient Finetuning via Random Orthogonal Subspace Projection (ROSA)',
    author='Marawan Gamal',
    license='MIT',
)
