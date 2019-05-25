# disable builtin and suffix rules
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

.PHONY: help clean deps lint test coverage docs release install jenkins

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "clean-env - remove env"
	@echo "env - set up virtualenv for the project"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "deps - force update of requirements specs"
	@echo "install - install the package to the active Python's site-packages"
	@echo "release - package and upload a release"

env:
	virtualenv --setuptools $@
	$@/bin/pip install -U "setuptools"
	$@/bin/pip install -U "pip"
	$@/bin/pip install -U "pip-tools"

# sentinel file to ensure installed requirements match current specs
env/.requirements: requirements.txt requirements-test.txt docs/requirements-docs.txt | env
	$|/bin/pip-sync $^
	touch $@

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

# this is not part of the "clean" target
clean-env:
	rm -rf env/

clean-test:
	rm -f .coverage
	rm -fr htmlcov/

lint: env/.requirements
	env/bin/flake8 classification tests

test: env/.requirements
	nosetests --debug integration_tests

coverage: env/.requirements test
	env/bin/coverage run --source classification setup.py test
	env/bin/coverage report -m
	env/bin/coverage html
	open htmlcov/index.html

install: clean
	pip install --upgrade .

release: env/.requirements clean
	env/bin/fullrelease

deps:
	@touch requirements.in requirements-test.in
	$(MAKE) requirements.txt requirements-test.txt

requirements.txt requirements-test.txt: %.txt: %.in | env
	$|/bin/pip-compile --no-index $^
