from decouple import config
from qiskit import IBMQ
from datetime import datetime

def pdt():
    print(datetime.now())

def interim_result_callback(job_id, interim_result):
    print(f"interim result: {interim_result}")

IBMQ.save_account(config('IBM_API_KEY'), overwrite=True)

IBMQ.load_account()
IBMQ.providers()
provider = IBMQ.get_provider(hub='strangeworks-hub', group='science-team', project='science-test')
print(provider.backends())

if provider.has_service('runtime'):
    pdt()
    provider.runtime.pprint_programs()
    program = provider.runtime.program('sample-program')
    print(program)
    backend = provider.get_backend('ibmq_montreal')
    program_inputs = {
        'iterations': 3
    }
    options = {'backend_name': backend.name()}
    job = provider.runtime.run(program_id="sample-program",
                            options=options,
                            inputs=program_inputs,
                            callback=interim_result_callback
                            )
    print(f"job id: {job.job_id()}")
    result = job.result()
    print(result)