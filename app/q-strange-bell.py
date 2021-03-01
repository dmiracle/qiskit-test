import strangeworks.qiskit
import qiskit

print("Hello world! Welcome to QuantumComputing.com!")

qc = qiskit.QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])


backend = strangeworks.qiskit.get_backend("qasm_simulator")
job = qiskit.execute([qc], backend, shots=100)