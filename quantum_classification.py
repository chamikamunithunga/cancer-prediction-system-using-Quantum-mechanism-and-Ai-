"""
WHAT THIS PROGRAM DOES:
----------------------
This is a quantum machine learning program that teaches a quantum computer 
to classify data into two groups. It's like teaching a computer to recognize 
patterns, but using quantum physics instead of regular computer logic.

The program creates some test data, builds a quantum learning system, trains 
it to recognize patterns, and then tests how well it learned.

WHY YOU MADE THIS:
------------------
You made this to:
- Learn how quantum computers can be used for machine learning
- Understand quantum computing in a practical way
- Explore new technology that might be important in the future
- Get hands-on experience with quantum programming

THE VALUE OF THIS PROGRAM:
--------------------------
This program is valuable because:

1. It teaches you quantum machine learning - a cutting-edge field that 
   combines quantum physics with artificial intelligence

2. It's a working example you can learn from and modify for your own projects

3. It demonstrates real quantum computing concepts in action, not just theory

4. Quantum machine learning might be the future - some problems that are 
   hard for regular computers might be easier for quantum computers

5. It's a foundation you can build on - you can adapt this for real problems 
   like recognizing images, analyzing data, or finding patterns

In simple terms: This program shows you how to use quantum computers for 
learning and classification, which is a skill that could be very valuable as 
quantum technology becomes more common.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_aer import AerSimulator
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time

# Try to use IBM Quantum, fallback to local simulator if authentication fails
use_ibm = False
backend = None
sampler_backend = None

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
    service = QiskitRuntimeService()
    # Try to get a simulator first (faster and free)
    try:
        backend = service.get_backend("ibmq_qasm_simulator")
        sampler_backend = Sampler(backend=backend)
        use_ibm = True
        print(f"✓ Connected to IBM Quantum backend: {backend.name}")
    except:
        # If simulator not available, get the least busy real backend
        print("IBM simulator not found, trying real hardware...")
        backend = service.least_busy(simulator=True, operational=True)
        sampler_backend = Sampler(backend=backend)
        use_ibm = True
        print(f"✓ Connected to IBM Quantum backend: {backend.name}")
except Exception as e:
    print(f"⚠ Could not connect to IBM Quantum: {str(e)}")
    print("Using local simulator instead (no authentication required)...")
    # Use local StatevectorSampler as fallback (more compatible with VQC)
    from qiskit.primitives import StatevectorSampler
    backend = AerSimulator()
    sampler_backend = StatevectorSampler()
    use_ibm = False
    print(f"✓ Using local simulator: {backend.name}")

print()

# Generate a simple 2-class classification dataset
# This creates a dataset that's hard for classical methods but good for quantum
feature_dim = 2  # 2 features
training_size = 40  # More data for better learning
test_size = 20

# Generate ad-hoc dataset (designed for quantum classification)
# Smaller gap makes classification easier
sample_total, training_input, test_input, class_labels = ad_hoc_data(
    training_size=training_size,
    test_size=test_size,
    n=feature_dim,
    gap=0.2,  # Smaller gap = easier classification
    plot_data=False
)

print(f"\nDataset created:")
print(f"  Training samples per class: {training_size}")
print(f"  Test samples per class: {test_size}")
print(f"  Features per sample: {feature_dim}")
print(f"  Total training samples: {len(training_input)}")
print(f"  Total test samples: {len(test_input)}")


X_train = np.array(training_input)
X_test = np.array(test_input)


y_train = np.array([0] * training_size + [1] * training_size)


y_test = np.argmax(class_labels, axis=1)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create the quantum feature map
# This maps classical data into quantum states
# Increased reps for more expressive feature mapping
feature_map = zz_feature_map(feature_dimension=feature_dim, reps=4, entanglement="linear")
print(f"\nFeature map: {feature_map.num_qubits} qubits, reps=4")

# Create the variational quantum circuit (ansatz)
# This is the trainable part of the quantum classifier
# Increased reps for more trainable parameters
ansatz = real_amplitudes(num_qubits=feature_dim, reps=5)
print(f"Ansatz: {ansatz.num_qubits} qubits, {ansatz.num_parameters} parameters, reps=5")

# Create a custom optimizer wrapper to track progress
class ProgressTracker:
    def __init__(self, base_optimizer):
        self.base_optimizer = base_optimizer
        self.history = []
        self.iteration = 0
        
    def minimize(self, fun, x0, jac=None, bounds=None):
        """Wrapper to track optimization progress"""
        def tracked_fun(x):
            result = fun(x)
            self.iteration += 1
            self.history.append(result)
            # Show progress every 25 iterations for better monitoring
            if self.iteration % 25 == 0 or self.iteration == 1:
                print(f"  Iteration {self.iteration:3d}: Loss = {result:.6f}", end='')
                if self.iteration > 1:
                    improvement = self.history[0] - result
                    pct = (improvement / self.history[0]) * 100 if self.history[0] > 0 else 0
                    print(f" (↓{pct:.1f}% from start)")
                else:
                    print()
            return result
        
        # Use the base optimizer's minimize method
        return self.base_optimizer.minimize(tracked_fun, x0, jac=jac, bounds=bounds)

# Create the Variational Quantum Classifier (VQC)
# COBYLA optimizer with more iterations for better convergence
base_optimizer = COBYLA(maxiter=500)
progress_tracker = ProgressTracker(base_optimizer)
optimizer = progress_tracker
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler_backend
)

print("\n" + "="*50)
print("Training the quantum classifier...")
print("="*50)
print("Monitoring training progress (showing every 50 iterations)...")
print()

start_time = time.time()

# Train the classifier
vqc.fit(X_train, y_train)

training_time = time.time() - start_time

print()
print(f"Training completed in {training_time:.2f} seconds!")
if len(progress_tracker.history) > 0:
    print(f"Total iterations: {len(progress_tracker.history)}")
    if len(progress_tracker.history) > 1:
        print(f"Initial loss: {progress_tracker.history[0]:.6f}")
        print(f"Final loss: {progress_tracker.history[-1]:.6f}")
        improvement = progress_tracker.history[0] - progress_tracker.history[-1]
        print(f"Loss improvement: {improvement:.6f} ({improvement/progress_tracker.history[0]*100:.1f}% reduction)")
    else:
        print(f"Final loss: {progress_tracker.history[0]:.6f}")

# Test the classifier
print("\n" + "="*50)
print("Testing the quantum classifier...")
print("="*50)

test_score = vqc.score(X_test, y_test)
print(f"\nTest accuracy: {test_score:.2%}")

# Make predictions
predictions = vqc.predict(X_test)
print(f"\nPredictions: {predictions}")
print(f"True labels:  {y_test}")

# Show some example predictions
print("\n" + "="*50)
print("Example predictions:")
print("="*50)
for i in range(min(5, len(X_test))):
    pred = predictions[i]
    actual = y_test[i]
    correct = "✓" if pred == actual else "✗"
    print(f"Sample {i+1}: Predicted={pred}, Actual={actual} {correct}")

print("\n" + "="*50)
print("Quantum classification complete!")
print("="*50)

