Flojoy - visual python function

## Publish Pakcage on PYPI
### Uploading file via *Twine*
You have to install `twine` package first if not installed already.
To install `twine` run following command: 
  `pip install twine`
  
- Update version in [setup.py](setup.py#L5). For example: `version = '0.0.1'` for prod version and `version = '0.0.1-dev'` for dev version release,
- Run following command to make a distribution file:
  `python3 setup.py sdist`
- To upload files on `PYPI`, run following command: 
`twine upload dist/*`

**Note:** You'll need `username` and `password` to make a release. Please get in touch with the team for credentials.
