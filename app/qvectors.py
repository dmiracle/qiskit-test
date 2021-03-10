import qiskit as qs

qc = qs.QuantumCircuit(1)
qc.initialize([1,0],0)
qc.draw()