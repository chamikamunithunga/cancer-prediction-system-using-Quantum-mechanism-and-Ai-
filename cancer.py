import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
import time

# 1. සැබෑ දත්ත ලබා ගැනීම (Load Real Data)
cancer_data = load_breast_cancer()
X = cancer_data.data  # සෛල මිනුම් 30ක්
y = cancer_data.target # 0 = Malignant, 1 = Benign

# 2. දත්තවල ප්‍රමාණය අඩු කිරීම (Dimensionality Reduction)
# Qubits 2කට ගැලපෙන සේ වැදගත්ම දත්ත 2ක් පමණක් ගනී
X_reduced = PCA(n_components=2).fit_transform(X)
X_scaled = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_reduced)

# පුහුණු කිරීම සඳහා දත්ත බෙදා ගැනීම (Train-Test Split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. ක්වොන්ටම් පරිපථය නිර්මාණය (Quantum Circuit Setup)
num_qubits = 2
feature_map = zz_feature_map(feature_dimension=num_qubits, reps=2, entanglement="linear")
ansatz = real_amplitudes(num_qubits=num_qubits, reps=3)

# 4. ක්වොන්ටම් බුද්ධිය පුහුණු කිරීම (VQC Training)
# Try to use IBM Quantum, fallback to local simulator if authentication fails
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
    from qiskit_aer import AerSimulator
    service = QiskitRuntimeService()
    # Get available backends
    backends = service.backends()
    if len(backends) > 0:
        # Try to get any available simulator first
        simulators = [b for b in backends if 'simulator' in b.name.lower() or 'sim' in b.name.lower()]
        if simulators:
            backend = simulators[0]
            sampler = Sampler(backend=backend)
            print(f"✓ IBM Quantum සම්බන්ධ විය: {backend.name}")
        else:
            # If no simulator, try to get least busy operational backend
            try:
                backend = service.least_busy(operational=True)
                sampler = Sampler(backend=backend)
                print(f"✓ IBM Quantum සම්බන්ධ විය: {backend.name}")
            except:
                # If that fails, use first available backend
                backend = backends[0]
                sampler = Sampler(backend=backend)
                print(f"✓ IBM Quantum සම්බන්ධ විය: {backend.name}")
    else:
        raise Exception("No backends available")
except Exception as e:
    print(f"⚠ IBM Quantum සම්බන්ධ වීමට නොහැකි විය: {str(e)}")
    print("Local simulator භාවිතා කරමින්...")
    sampler = StatevectorSampler()

optimizer = COBYLA(maxiter=100)

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler
)

print("ක්වොන්ටම් පද්ධතිය සැබෑ පිළිකා දත්ත මගින් පුහුණු වෙමින් පවතී...")
start_time = time.time()
vqc.fit(X_train, y_train)
end_time = time.time()

# 5. ප්‍රතිඵල පරීක්ෂා කිරීම (Evaluation)
accuracy = vqc.score(X_test, y_test)
print(f"\n--- විනිශ්චය වාර්තාව ---")
print(f"පුහුණු වීමට ගතවූ කාලය: {end_time - start_time:.2f} seconds")
print(f"නිවැරදිතාවය (Accuracy): {accuracy * 100:.2f}%")

# උදාහරණයක් ලෙස එක් රෝගියෙකු පරීක්ෂා කිරීම
sample_pred = vqc.predict(X_test[:1])
# Convert to numpy array and get first element
pred_value = np.array(sample_pred).flatten()[0]
status = "පිළිකා අවදානමක් ඇත (Malignant)" if pred_value == 0 else "පිළිකා අවදානමක් නැත (Benign)"
print(f"පරීක්ෂා කළ රෝගියාගේ තත්ත්වය: {status}")