### al-template-engine

The template of alchemist model engine


#### Pre-requisite

* PyEnv - python version management (nvm)
 * using `.python-version` to specify the repo python version(`anaconda3-5.3.1`)
 * `pyenv install`
* PipEnv - python package management (yarn) / conda - anaconda package management
 * `brew install pipenv`
 * using `Pipfile` to specify python compiler version and packages
 * `pipenv --three` (generating Pipfile.lock) / `conda install --file Pipfile`
 * `pipenv shell` to activate the virtual environment if needed
* IDE with linting, formatting, REPL (Sublime Text with plugins)
 * open the corresponding python version REPL under `Tools > SublimeREPL > Python`
 * evaluate selection or file using your local shortcuts
 * SublimeText Plugins
  * SublimeREPL
  * SUblimeLinter?
* pip - the default python package manager (npm)
 * not recommended for local development

#### Development
* Linting
 * how to incorporate flake8 into local and ci env?
 * how to config flake8
 * commitlint?
* Formating - [black](https://github.com/python/black)
* Testing - pytest
* Semver

#### Training
use a terminal service to connecting to remote GPU pool if possible

#### Deploy
