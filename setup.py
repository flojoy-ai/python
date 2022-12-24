from distutils.core import setup
setup(
    name='joyflo-sma',
    packages=['joyflo'],
    version='0.3.5',
    license='MIT',
    description='Some description',
    author='flojoy',
    author_email='example@email.com',
    url='https://github.com/flojoy-io/flojoy-python',
    download_url='https://github.com/flojoy-io/flojoy-python/archive/refs/heads/main.zip',
    keywords=['flojoy', 'visual', 'python-visual'],
    install_requires=[
        'python-box',
        'networkx',
        'numpy',
        'redis',
        'rq',
        'scipy',
        'pytest',
        'python-dotenv'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
