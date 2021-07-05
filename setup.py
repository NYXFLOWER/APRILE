import re
from io import open
from os import path

from setuptools import setup

# read the contents of requirements.txt
requirements = [
	"scikit-learn>=0.21.2",
	"scipy",
	"matplotlib",
	"networkx>=2.4",
	"pandas",
	"numpy",
	"goatools",
	"cython"
]

# auto-detect package version
def get_version(*file_paths):
	with open(path.join(path.dirname(__file__), *file_paths), encoding='utf8') as f:
		version_file = f.read()

	version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
	if version_match:
		return version_match.group(1)
	raise RuntimeError("Unable to find version info.")

readme = open("README.md").read()
version = get_version("aprile", "__init__.py")

# Run the setup
setup(
	name="aprile",
	version=version,
	description="Adverse Polypharmacy Reaction Intelligent Learner and Explainer (APRILE) -- An explainable machine learning framework for exploring the molecular mechanisms of drug side effects (including disease, symptoms and mentel disorders).",
	long_description=readme,	
	long_description_content_type="text/markdown",
	author="Hao Xu",
	url="https://github.com/NYXFLOWER/APRILE",
	author_email="nyx0flower@gmail.com",
	project_urls={
		"Bug Tracker": "https://github.com/NYXFLOWER/APRILE/issues",
		"Documentation": "https://aprile.readthedocs.io",
		"Source": "https://github.com/NYXFLOWER/APRILE/aprile",
	},
	packages=['aprile'],
	python_requires=">=3.7",
	install_requires=requirements,
	setup_requires=["setuptools>=57.0.0"],
	license="MIT License",
	keywords="machine learning, bioinformatics, graph neural network, adverse drug reaction, disease mechanism",
	include_package_data=True,
	package_data={
		'': ['*.pt', '*.pkl']
	},
	data_files=[
		('', ['aprile/data.pkl', 'aprile/POSE-pred.pt', 'aprile/data_dict.pkl'])
	],
	classifiers=[
		"Intended Audience :: Developers",
		"Intended Audience :: Education",
		"Intended Audience :: Healthcare Industry",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Scientific/Engineering :: Medical Science Apps.",
		"Natural Language :: English",
	],
)