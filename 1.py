from qiskit_ibm_runtime import QiskitRuntimeService

# Replace with your NEW private token
# Running this once saves it to your computer's local config file
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="J9MfPXBTm9Ivy75rfMz6bUmLDUFe5TeUQ1FaGVLN8Kqx", overwrite=True)