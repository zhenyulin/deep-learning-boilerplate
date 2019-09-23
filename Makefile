install:
	@pipenv install --python=`pyenv which python`

shell:
	@pipenv shell

cleanup:
	@pipenv clean

test:
	@pytest test --cov=src
