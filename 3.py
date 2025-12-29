from qiskit_ibm_runtime import QiskitRuntimeService

# Load your saved account
service = QiskitRuntimeService()

# Automatically find the least busy real quantum computer
backend = service.least_busy(simulator=False, operational=True)
print(f"Connected to real hardware: {backend.name}")

# Now, any circuit you run will be sent to a real chip in an IBM lab