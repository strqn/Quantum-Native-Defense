# EXPERIMENT 2: ADAPTIVE THRESHOLD CALIBRATION ON MULTIPLE IBM QUANTUM DEVICES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# CONFIGURATION
IBM_API_TOKEN = "8DSRkWd_Q7I49IpxLgUFT8l2G97GswTwTng8iLs58RI1" 
BACKEND_NAMES = ["ibm_torino", "ibm_marrakesh", "ibm_fez"]
N_SHOTS = 4096
N_REPEATS = 5
N_PAIRS = 5
C_VALUE = 1.645         # Standard deviation multiplier (95% CI)

CANDIDATE_PAIRS = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)]


# STEP 1: CONNECT TO IBM QUANTUM

def connect_ibm(token):
    """Initialize IBM Quantum service."""
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        overwrite=True
    )
    return QiskitRuntimeService(channel="ibm_quantum_platform")


# STEP 2: BELL STATE PREPARATION CIRCUIT (with named classical register)

def bell_pair_circuit():
    """
    Create a 2-qubit circuit that prepares |Phi+> = (|00>+|11>)/sqrt(2).
    Uses a classical register named 'bell' for reliable access.
    """
    qr = QuantumRegister(2)
    cr = ClassicalRegister(2, 'bell')
    qc = QuantumCircuit(qr, cr)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, cr[0])
    qc.measure(1, cr[1])
    return qc


# STEP 3: FIDELITY ESTIMATION FROM 2-BIT COUNTS

def fidelity_from_counts(counts):
    """
    Estimate fidelity between the prepared state and |Phi+>.
    Expects counts dictionary with keys '00', '01', '10', '11'.
    """
    shots = sum(counts.values())
    p00 = counts.get('00', 0) / shots
    p11 = counts.get('11', 0) / shots
    return (np.sqrt(p00) + np.sqrt(p11))**2 / 2


# STEP 4: SAFELY EXTRACT DEVICE CALIBRATION DATA

def get_device_calibration(backend):
    """Extract median T2, readout error, and CX error from backend properties."""
    props = backend.properties()
    if not props:
        return {}

    t2_vals = []
    for q in range(backend.num_qubits):
        try:
            t2 = props.qubit_property(q, 'T2')
            if t2 is not None:
                t2_vals.append(t2[0])
        except:
            pass
    t2_median = np.median(t2_vals) if t2_vals else np.nan

    ro_vals = []
    for q in range(backend.num_qubits):
        try:
            ro = props.qubit_property(q, 'readout_error')
            if ro is not None:
                ro_vals.append(ro[0])
        except:
            pass
    ro_median = np.median(ro_vals) if ro_vals else np.nan

    cx_err_vals = []
    for gate in props.gates:
        if gate.gate == 'cx':
            for param in gate.parameters:
                if param.name == 'gate_error':
                    cx_err_vals.append(param.value)
    cx_median = np.median(cx_err_vals) if cx_err_vals else np.nan

    return {
        'T2_median_us': t2_median * 1e6 if not np.isnan(t2_median) else None,
        'readout_error': ro_median,
        'cx_error': cx_median
    }


# STEP 5: RUN CALIBRATION ON A SINGLE BACKEND (NAMED REGISTER)

def calibrate_backend(backend, pairs, shots, repeats):
    """
    Run Bell pair circuits on a given backend using explicit initial_layout
    and the fixed classical register name 'bell'.
    """
    sampler = Sampler(mode=backend)
    all_fids = []

    for pair in pairs:
        q0, q1 = pair
        qc = bell_pair_circuit()
        transpiled = transpile(
            qc,
            backend=backend,
            initial_layout=[q0, q1],
            optimization_level=3
        )
        print(f"   Pair ({q0},{q1}) - depth: {transpiled.depth()}, gates: {transpiled.size()}")

        for rep in range(repeats):
            job = sampler.run([transpiled], shots=shots)
            result = job.result()
            # Direct access using the fixed register name 'bell'
            counts = result[0].data.bell.get_counts()
            fid = fidelity_from_counts(counts)
            all_fids.append(fid)
            print(f"      Rep {rep+1}: F = {fid:.5f}")

    all_fids = np.array(all_fids)
    eta = 1.0 - np.mean(all_fids)
    sigma_s = np.std(all_fids, ddof=1)
    tau = 1.0 - eta - C_VALUE * sigma_s
    return eta, sigma_s, tau, all_fids


# STEP 6: VALIDATION WITH NAMED CLASSICAL REGISTER

def validation_test(backend, tau, shots=1024):
    """
    Test discrimination: intact watermark vs bit-flip tampering.
    Uses a classical register named 'wm'.
    """
    # ----- Intact: watermark on qubit 0 (virtual) -----
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1, 'wm')
    qc_intact = QuantumCircuit(qr, cr)
    qc_intact.h(0)
    qc_intact.p(0.3 * np.pi, 0)
    qc_intact.measure(0, cr[0])
    transp_intact = transpile(qc_intact, backend=backend, optimization_level=3)

    # ----- Tampered: add X gate -----
    qc_tamper = QuantumCircuit(qr, cr)
    qc_tamper.h(0)
    qc_tamper.p(0.3 * np.pi, 0)
    qc_tamper.x(0)
    qc_tamper.measure(0, cr[0])
    transp_tamper = transpile(qc_tamper, backend=backend, optimization_level=3)

    sampler = Sampler(mode=backend)
    job_intact = sampler.run([transp_intact], shots=shots)
    job_tamper = sampler.run([transp_tamper], shots=shots)

    # Direct access using fixed register name 'wm'
    counts_intact = job_intact.result()[0].data.wm.get_counts()
    counts_tamper = job_tamper.result()[0].data.wm.get_counts()

    # Fidelity to ideal state |+> rotated by P(0.3pi).
    def fidelity_single(counts):
        shots = sum(counts.values())
        p0 = counts.get('0', 0) / shots
        p1 = counts.get('1', 0) / shots
        return (np.sqrt(p0 * 0.5) + np.sqrt(p1 * 0.5))**2

    F_intact = fidelity_single(counts_intact)
    F_tamper = fidelity_single(counts_tamper)

    return F_intact, F_tamper


# MAIN EXECUTION

def main():
    print("=" * 80)
    print("ADAPTIVE THRESHOLD CALIBRATION ON IBM QUANTUM ")
    print("=" * 80)

    service = connect_ibm(IBM_API_TOKEN)

    backends = {}
    for name in BACKEND_NAMES:
        try:
            backend = service.backend(name)
            backends[name] = backend
            print(f"\nConnected to {name}")
            print(f"   Qubits: {backend.num_qubits}")
            status = backend.status()
            print(f"   Status: {status.status_msg}, pending jobs: {status.pending_jobs}")
            cal = get_device_calibration(backend)
            if cal:
                if cal['T2_median_us']:
                    print(f"   T2 (median): {cal['T2_median_us']:.2f} us")
                if cal['readout_error']:
                    print(f"   Readout error (median): {cal['readout_error']:.5f}")
                if cal['cx_error']:
                    print(f"   CX error (median): {cal['cx_error']:.5f}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

    if not backends:
        print("No backends available. Exiting.")
        return

    results = {}
    all_fidelities = {}

    for name, backend in backends.items():
        print(f"\n{'=' * 60}")
        print(f"CALIBRATING: {name}")
        print('=' * 60)

        num_qubits = backend.num_qubits
        valid_pairs = [pair for pair in CANDIDATE_PAIRS if pair[0] < num_qubits and pair[1] < num_qubits]
        if len(valid_pairs) < N_PAIRS:
            print(f"Note: Only {len(valid_pairs)} valid pairs, reducing N_PAIRS to {len(valid_pairs)}")
        pairs = valid_pairs[:N_PAIRS]

        eta, sigma_s, tau, fids = calibrate_backend(backend, pairs, N_SHOTS, N_REPEATS)
        results[name] = {
            'eta': eta,
            'sigma_s': sigma_s,
            'tau': tau,
            'pairs': pairs,
            'n_qubits': backend.num_qubits,
            'n_runs': len(fids)
        }
        all_fidelities[name] = fids

        print(f"\nRESULTS for {name}:")
        print(f"   eta (noise floor) = {eta:.5f}")
        print(f"   sigma_s = {sigma_s:.5f}")
        print(f"   tau (threshold) = {tau:.5f}")

    print("\n" + "=" * 80)
    print("SUMMARY: ADAPTIVE THRESHOLDS")
    print("=" * 80)
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            'Device': name,
            'Qubits': res['n_qubits'],
            'Pairs': str(len(res['pairs'])),
            'eta (noise)': f"{res['eta']:.5f}",
            'sigma_s': f"{res['sigma_s']:.5f}",
            'tau': f"{res['tau']:.5f}"
        })
    df_summary = pd.DataFrame(summary_data)
    display(df_summary)
    df_summary.to_csv('adaptive_threshold_calibration.csv', index=False)
    print("\nCalibration results saved to 'adaptive_threshold_calibration.csv'")

    print("\n" + "=" * 80)
    print("VALIDATION: INTACT VS. TAMPERED DATA")
    print("=" * 80)
    val_results = []
    for name, backend in backends.items():
        tau = results[name]['tau']
        print(f"\nTesting {name} ...")
        
        # Run validation
        F_intact, F_tamper = validation_test(backend, tau, shots=2048)
        
        # Save raw results only - no decision logic
        val_results.append({
            'Device': name,
            'tau': f"{tau:.5f}",
            'F_intact': f"{F_intact:.5f}",
            'F_tampered': f"{F_tamper:.5f}"
        })
        print(f"   tau        = {tau:.5f}")
        print(f"   F_intact   = {F_intact:.5f}")
        print(f"   F_tampered = {F_tamper:.5f}")

    df_val = pd.DataFrame(val_results)
    display(df_val)
    df_val.to_csv('threshold_validation.csv', index=False)

    # Plotting histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, fids) in zip(axes, all_fidelities.items()):
        ax.hist(fids, bins=10, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(1 - results[name]['eta'], color='red', linestyle='--', label=f"1-eta")
        ax.axvline(results[name]['tau'], color='green', linestyle='-', label=f"tau")
        ax.set_title(name)
        ax.set_xlabel('Fidelity')
        ax.set_ylabel('Count')
        ax.legend()
    plt.tight_layout()
    plt.savefig('threshold_calibration_histograms.svg', format='svg', dpi=300)
    plt.show()

    print("\nExperiment complete.")

if __name__ == "__main__":
    main()
  
