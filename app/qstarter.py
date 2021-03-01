# Modify qiskit textbook code from chapter 1.2.3 to run on Strangeworks
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram

n = 8 
n_q = n 
n_b = n
qc_output = QuantumCircuit(n_q, n_b)

for j in range(n):
    qc_output.measure(j,j)

# We are going to draw any circuit we can so don't explicitly call draw
qc_output.draw()

sim = Aer.get_backend('qasm_simulator')

qobj = assemble(qc_output)
count = sim.run(qobj).result().get_counts()
plot_histogram(count)