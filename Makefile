env:
	conda env update --file environment.yml --prune

html:
	myst build --html

clean:
	rm -rf figures audio _build

