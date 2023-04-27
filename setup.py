from distutils.core import setup
setup(
    name='flojoy',
    packages=['flojoy'],
    version='0.1.3-dev',
    license='MIT',
    description='Python client library for Flojoy.',
    author='flojoy',
    author_email='jack.parmer@proton.me',
    url='https://github.com/flojoy-io/flojoy-python',
    download_url='https://github.com/flojoy-io/flojoy-python/archive/refs/heads/main.zip',
    keywords=['data-acquisition', 'lab-automation', 'low-code', 'python', 'scheduler', 'topic'],
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
        'Programming Language :: Python :: 3.10',
    ],
)
