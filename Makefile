install:
	@pipenv --three
	@pipenv shell

cleanup:
	@pipenv clean

test:
	@pytest test --cov=src
