# Qiskit Test

Starter python project using qikit. The goal of this project is to have a properly structured python project that uses qiskit and can be used to chase down how various things work.

## Make environments 
Run `make venv` to create a new python virtual environment. To add new python packages add the package to the `requirements.txt` file and run `make update`. To delete the virtual environment run `make clean`.

## `.env` file for keys
The python `decouple` library is used to load the IBM api key. Create a file in the root of this repo called `.env` and add your key to the file as
```
IBM_API_KEY=<your_key_here>
```
## Notebooks

- `qiskit.ipynb` many random qiskit tests
- `ibmq.ipynb` run circuits on ibm hardware
- `requests.ipynb` directly call the runtime api
- `vqe.ipynb` vqe 3 ways: local, standard hardware, runtime

