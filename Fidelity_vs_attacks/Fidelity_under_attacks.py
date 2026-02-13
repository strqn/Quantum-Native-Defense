# QUANTUM WATERMARKING ON REAL IBM QUANTUM HARDWARE

from google.colab import drive
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.circuit.library import Initialize
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION

class Config:
    
    IBM_API_TOKEN = ""
    IBM_BACKEND = "ibm_torino"

    IMAGE_PATH = "/content/drive/MyDrive/IM-0125-0001.jpeg"
    IMAGE_SIZE = (128, 128)
    REGION_SIZE = (32, 32)  
    PHASE_SHIFT_FACTOR = 0.3
    NUM_SHOTS = 1024 
    OUTPUT_PATH = '/content/drive/MyDrive/quantum_watermark_ibm_results.csv'


# IBM QUANTUM SERVICE SETUP

def setup_ibm_quantum(api_token, backend_name):
    """
    Initialize IBM Quantum service and get backend

    Args:
        api_token: Your IBM Quantum API token
        backend_name: Name of quantum backend (e.g., 'ibm_torino')

    Returns:
        Backend object
    """
    try:
        # Save credentials
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=api_token,
            overwrite=True
        )
        print("IBM Quantum credentials saved")

        # Initialize service
        service = QiskitRuntimeService(channel="ibm_quantum_platform")

        # Get backend
        backend = service.backend(backend_name)

        print(f"Connected to: {backend.name}")
        print(f"   Qubits: {backend.num_qubits}")
        print(f"   Status: {backend.status().status_msg}")
        print(f"   Pending jobs: {backend.status().pending_jobs}")

        return service, backend

    except Exception as e:
        print(f"Error connecting to IBM Quantum: {e}")
        print("\nMake sure to:")
        print("   1. Replace 'YOUR_IBM_QUANTUM_API_TOKEN_HERE' with your actual token")
        print("   2. Get your token from: https://quantum.ibm.com/")
        print("   3. Check you have access to ibm_torino on your plan")
        return None, None


# STEP 1: MOUNT GOOGLE DRIVE

def mount_drive():
    """Mount Google Drive safely"""
    try:
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")
    except Exception as e:
        print(f"Drive already mounted or error: {e}")


# STEP 2: LOAD AND PREPROCESS IMAGE

def load_and_preprocess_image(path, size):
    """Load image, convert to grayscale, resize and normalize"""
    try:
        img = Image.open(path).convert("L")
        img = img.resize(size)
        img_array = np.array(img) / 255.0
        print(f"Image loaded successfully. Shape: {img_array.shape}")
        return img_array
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def visualize_image(img_array, title="Image", cmap='gray'):
    """Display image with matplotlib"""
    plt.figure(figsize=(6, 6))
    plt.imshow(img_array, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.show()


# STEP 3: EXTRACT REGION

def extract_region(img_array, region_size):
    """Extract small region from image for quantum processing"""
    h, w = region_size
    region = img_array[0:h, 0:w]
    print(f"Region extracted: {region.shape}")
    return region


# STEP 4: QUANTUM STATE PREPARATION CIRCUIT

def create_state_preparation_circuit(region):
    """
    Create quantum circuit for state preparation

    Args:
        region: 2D numpy array (must be 2^n size for n qubits)

    Returns:
        Quantum circuit with initial state
    """
    flat_region = region.flatten()
    n_qubits = int(np.log2(len(flat_region)))

    # Normalize
    norm = np.linalg.norm(flat_region)
    if norm < 1e-10:
        flat_region += 1e-10
        norm = np.linalg.norm(flat_region)

    normalized = flat_region / norm

    # Create circuit
    qc = QuantumCircuit(n_qubits)

    # Initialize state
    qc.initialize(normalized, range(n_qubits))

    print(f"State preparation circuit created ({n_qubits} qubits)")
    return qc, normalized


# STEP 5: WATERMARK EMBEDDING CIRCUIT

  
def generate_watermark_pattern(n_qubits, pattern_type='alternating'):
    """Generate binary watermark pattern for qubits"""
    if pattern_type == 'alternating':
        pattern = [i % 2 for i in range(n_qubits)]
    elif pattern_type == 'random':
        np.random.seed(42)
        pattern = np.random.randint(0, 2, n_qubits).tolist()
    else:
        pattern = [1] * n_qubits

    return pattern

def embed_watermark_circuit(state_prep_circuit, phase_factor, pattern_type='alternating'):
    """
    Embed quantum watermark using phase gates

    Args:
        state_prep_circuit: Initial state preparation circuit
        phase_factor: Strength of phase shift
        pattern_type: Type of watermark pattern

    Returns:
        Circuit with watermark embedded
    """
    n_qubits = state_prep_circuit.num_qubits
    watermark_bits = generate_watermark_pattern(n_qubits, pattern_type)

    # Create watermark circuit
    watermark_qc = state_prep_circuit.copy()
    watermark_qc.barrier()

    # Apply phase shifts based on watermark
    for qubit, bit in enumerate(watermark_bits):
        if bit == 1:
            watermark_qc.p(np.pi * phase_factor, qubit)

    watermark_qc.barrier()

    print(f"Watermark embedded with pattern: {watermark_bits}")
    print(f"   Phase factor: {phase_factor}")

    return watermark_qc, watermark_bits


# STEP 6: TAMPERING SIMULATION CIRCUIT

  
def apply_tampering_circuit(watermark_circuit, tampering_type='bitflip', intensity=0.3):
    """
    Add tampering operations to circuit

    Args:
        watermark_circuit: Watermarked circuit
        tampering_type: 'bitflip', 'phase', or 'depolarizing'
        intensity: Tampering intensity

    Returns:
        Tampered circuit
    """
    tampered_qc = watermark_circuit.copy()
    tampered_qc.barrier()
    n_qubits = tampered_qc.num_qubits

    if tampering_type == 'bitflip':
        # Apply X gates to some qubits
        for i in range(0, n_qubits, 3):  # Every 3rd qubit
            tampered_qc.x(i)
        print("Bit-flip tampering applied")

    elif tampering_type == 'phase':
        # Apply phase rotation
        for i in range(n_qubits):
            tampered_qc.p(intensity * np.pi, i)
        print(f"Phase rotation tampering applied (theta={intensity}pi)")

    elif tampering_type == 'depolarizing':
        # Apply random single-qubit gates
        np.random.seed(42)
        for i in range(n_qubits):
            if np.random.random() < intensity:
                gate_choice = np.random.choice(['x', 'y', 'z'])
                if gate_choice == 'x':
                    tampered_qc.x(i)
                elif gate_choice == 'y':
                    tampered_qc.y(i)
                else:
                    tampered_qc.z(i)
        print(f"Depolarizing tampering applied (p={intensity})")

    tampered_qc.barrier()
    return tampered_qc


# STEP 7: MEASURE ON REAL QUANTUM HARDWARE

def measure_circuit_on_hardware(circuit, backend, service, shots=1024):
    """
    Execute circuit on real IBM quantum hardware

    Args:
        circuit: Quantum circuit to execute
        backend: IBM backend
        service: QiskitRuntimeService instance
        shots: Number of measurements

    Returns:
        Measurement results
    """
    # Add measurements
    qc_with_measurement = circuit.copy()
    qc_with_measurement.measure_all()

    # Transpile for hardware
    print(f"Transpiling circuit for {backend.name}...")
    transpiled_qc = transpile(qc_with_measurement, backend=backend, optimization_level=3)

    print(f"   Original circuit depth: {circuit.depth()}")
    print(f"   Transpiled circuit depth: {transpiled_qc.depth()}")
    print(f"   Transpiled circuit gates: {transpiled_qc.size()}")

    # Run on real hardware
    print(f"Submitting job to {backend.name} (this may take several minutes)...")
    print(f"   Queue position: {backend.status().pending_jobs}")

    sampler = Sampler(backend)
    job = sampler.run([transpiled_qc], shots=shots)

    print(f"Job submitted! Job ID: {job.job_id()}")
    print("Waiting for results from quantum hardware...")

    result = job.result()

    print("Results received from quantum hardware ")

    return result, transpiled_qc


# STEP 8: COMPUTE FIDELITY FROM MEASUREMENTS


def compute_fidelity_from_counts(counts_original, counts_tampered):
    """
    Compute fidelity from measurement counts

    Args:
        counts_original: Counts from original watermarked circuit
        counts_tampered: Counts from tampered circuit

    Returns:
        Estimated fidelity
    """
    # Get probability distributions
    all_states = set(counts_original.keys()) | set(counts_tampered.keys())

    total_original = sum(counts_original.values())
    total_tampered = sum(counts_tampered.values())

    # Calculate overlap
    overlap = 0
    for state in all_states:
        p_orig = counts_original.get(state, 0) / total_original
        p_tamp = counts_tampered.get(state, 0) / total_tampered
        overlap += np.sqrt(p_orig * p_tamp)

    # Fidelity is square of overlap
    fidelity = overlap ** 2

    return fidelity

def compute_qwef(fidelity, sigma_cut=0.15):
    """Compute Quantum Watermark Error Function"""
    delta_F = 1 - fidelity
    QWEF = 1 - np.exp(-delta_F / sigma_cut)
    return QWEF

def interpret_results(F, QWEF, threshold_F=0.90, threshold_QWEF=0.5):
    """Interpret integrity check results"""
    if F >= threshold_F and QWEF <= threshold_QWEF:
        status = "INTACT - Image integrity verified"
    elif F >= 0.80:
        status = "MINOR MODIFICATION - Possible noise or compression"
    elif F >= 0.60:
        status = "MODERATE TAMPERING - Significant alterations detected"
    else:
        status = "SEVERE TAMPERING - Image authenticity compromised"

    return status


# STEP 9: VISUALIZATION

  
def visualize_results(F, QWEF, status):
    """Create comprehensive visualization of results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    metrics = ['Fidelity', 'QWEF']
    values = [F, QWEF]
    colors = ['#4CAF50' if F > 0.85 else '#FF9800' if F > 0.6 else '#F44336',
              '#4CAF50' if QWEF < 0.5 else '#FF9800' if QWEF < 0.8 else '#F44336']

    axes[0].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_title('Quantum Watermarking Metrics (Real IBM Hardware)',
                      fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    for i, v in enumerate(values):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

    # Status indicator
    axes[1].axis('off')
    axes[1].text(0.5, 0.6, 'INTEGRITY STATUS', ha='center', va='center',
                 fontsize=16, fontweight='bold')
    axes[1].text(0.5, 0.4, status, ha='center', va='center',
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].text(0.5, 0.2, f'Fidelity: {F:.4f}\nQWEF: {QWEF:.4f}',
                 ha='center', va='center', fontsize=11, family='monospace')
    axes[1].text(0.5, 0.05, 'Measured on Real Quantum Hardware',
                 ha='center', va='center', fontsize=10, style='italic', color='blue')

    plt.tight_layout()
    plt.show()

def plot_measurement_distributions(counts_original, counts_tampered):
    """Plot measurement distributions from both circuits"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Original watermarked
    states_orig = list(counts_original.keys())
    values_orig = list(counts_original.values())
    ax1.bar(range(len(states_orig)), values_orig, color='green', alpha=0.6)
    ax1.set_xlabel('Measurement Outcome')
    ax1.set_ylabel('Counts')
    ax1.set_title('Original Watermarked (Real Hardware)')
    ax1.set_xticks([])

    # Tampered
    states_tamp = list(counts_tampered.keys())
    values_tamp = list(counts_tampered.values())
    ax2.bar(range(len(states_tamp)), values_tamp, color='red', alpha=0.6)
    ax2.set_xlabel('Measurement Outcome')
    ax2.set_ylabel('Counts')
    ax2.set_title('Tampered (Real Hardware)')
    ax2.set_xticks([])

    plt.tight_layout()
    plt.show()


# STEP 10: SAVE RESULTS


def save_results(F, QWEF, status, backend_name, job_ids, output_path):
    """Save results to CSV file"""
    results = pd.DataFrame({
        'Metric': ['Fidelity', 'QWEF', 'Status', 'Backend', 'Job_IDs'],
        'Value': [f'{F:.6f}', f'{QWEF:.6f}', status, backend_name, str(job_ids)]
    })

    try:
        results.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


# MAIN EXECUTION PIPELINE


def main():
    """Main execution pipeline for real quantum hardware"""
    print("="*70)
    print("QUANTUM WATERMARKING ON REAL IBM QUANTUM HARDWARE")
    print("Backend: IBM Torino (Real Quantum Computer)")
    print("="*70)

    # Step 1: Setup IBM Quantum
    service, backend = setup_ibm_quantum(Config.IBM_API_TOKEN, Config.IBM_BACKEND)
    if backend is None:
        return

    # Step 2: Mount Drive
    mount_drive()

    # Step 3: Load Image
    img_array = load_and_preprocess_image(Config.IMAGE_PATH, Config.IMAGE_SIZE)
    if img_array is None:
        return
    visualize_image(img_array, "Original X-ray Image")

    # Step 4: Extract Region
    region = extract_region(img_array, Config.REGION_SIZE)
    visualize_image(region, f"Selected {Config.REGION_SIZE[0]}x{Config.REGION_SIZE[1]} Region")

    # Step 5: Create State Preparation Circuit
    state_prep_qc, _ = create_state_preparation_circuit(region)

    # Step 6: Embed Watermark
    watermarked_qc, watermark_bits = embed_watermark_circuit(
        state_prep_qc,
        phase_factor=Config.PHASE_SHIFT_FACTOR
    )

    # Step 7: Create Tampered Version
    tampered_qc = apply_tampering_circuit(watermarked_qc, tampering_type='bitflip')

    # Step 8: Run on Real Quantum Hardware
    print("\n" + "="*70)
    print("EXECUTING ON REAL QUANTUM HARDWARE")
    print("="*70)

    result_original, transpiled_orig = measure_circuit_on_hardware(
        watermarked_qc, backend, service, shots=Config.NUM_SHOTS
    )

    result_tampered, transpiled_tamp = measure_circuit_on_hardware(
        tampered_qc, backend, service, shots=Config.NUM_SHOTS
    )

    # Extract counts
    counts_original = result_original[0].data.meas.get_counts()
    counts_tampered = result_tampered[0].data.meas.get_counts()

    print(f"\nOriginal measurements: {len(counts_original)} unique states")
    print(f"Tampered measurements: {len(counts_tampered)} unique states")

    # Step 9: Compute Fidelity
    F = compute_fidelity_from_counts(counts_original, counts_tampered)
    QWEF = compute_qwef(F, sigma_cut=0.15)
    status = interpret_results(F, QWEF)

    # Step 10: Display Results
    print("\n" + "="*70)
    print("RESULTS FROM REAL QUANTUM HARDWARE")
    print("="*70)
    print(f"Backend:             {backend.name}")
    print(f"Qubits used:         {watermarked_qc.num_qubits}")
    print(f"Measurements:        {Config.NUM_SHOTS} shots")
    print(f"Fidelity (F):        {F:.6f}")
    print(f"QWEF:                {QWEF:.6f}")
    print(f"Status:              {status}")
    print("="*70)

    # Visualizations
    plot_measurement_distributions(counts_original, counts_tampered)
    visualize_results(F, QWEF, status)

    # Step 11: Save Results
    job_ids = {
        'original': result_original.job_id,
        'tampered': result_tampered.job_id
    }
    save_results(F, QWEF, status, backend.name, job_ids, Config.OUTPUT_PATH)

    return F, QWEF, status, result_original, result_tampered


# RUN MAIN PIPELINE

if __name__ == "__main__":
    F, QWEF, status, result_orig, result_tamp = main()
  
