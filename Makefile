venv:
	python3 -m venv venv
	. venv/bin/activate; python -m pip install -U pip setuptools wheel
	. venv/bin/activate; python -m pip install -Ur requirements.txt
update:
	. venv/bin/activate; python -m pip install -U pip setuptools wheel
	. venv/bin/activate; python -m pip install -Ur requirements.txt	
test: venv
	. venv/bin/activate; python -m pip install -r requirements-dev.txt
clean:
	rm -rf venv