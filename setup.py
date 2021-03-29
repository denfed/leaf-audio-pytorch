import setuptools

description = 'PyTorch LEAF: a PyTorch Port of Google Research Learnable Audio Frontend'

setuptools.setup(
    name='leaf-audio-pytorch',
    version='0.1.0',
    packages=setuptools.find_packages(include=['leaf_audio_pytorch']),
    description=description,
    long_description=description,
    url='https://github.com/denfed/leaf-audio-pytorch',
    author='Dennis Fedorishin',
    author_email='dcfedori@buffalo.edu',
    install_requires=['torch>=1.8.0',
                      'numpy>=1.18.2'],
    python_requires='>=3.6',
    license='Apache 2.0',
    keywords='PyTorch learnable frontend audio port'
)