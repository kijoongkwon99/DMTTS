import os
import re
import setuptools


def get_requirements(req_path: str):
    with open(req_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

INSTALL_REQUIRES = get_requirements("requirements.txt")

def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()

def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, 'src', 'dmtts', '__init__.py')
    with open(version_file, encoding='utf-8') as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)

def get_author():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(current_dir, 'src', 'dmtts', '__init__.py')
    with open(init_file, encoding='utf-8') as f:
        return re.search(r'^__author__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)

def get_license():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    init_file = os.path.join(current_dir, 'src', 'dmtts', '__init__.py')
    with open(init_file, encoding='utf-8') as f:
        return re.search(r'^__license__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


setuptools.setup(
    name='dmtts',
    version=get_version(),
    author=get_author(),
    author_email='kijoongkwon@kaist.ac.kr',
    license=get_license(),
    description="DMTTS: VITS-based structure with ConvNeXt V2",
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/kijoongkwon99/DMTTS',
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(where='src', include=['dmtts*']),
    package_dir={'': 'src'},
    package_data={
        'dmtts.model.text': [
            'opencpop-strict.txt', 'wiktionary-23-7-2022-clean.tsv', 'cmudict.rep', 'cmudict_cache.pickle'
        ],
    },
    include_package_data=True,
)