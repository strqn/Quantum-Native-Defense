
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
import warnings
warnings.filterwarnings('ignore')


from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import Initialize


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T


import pandas as pd
from tqdm import tqdm

# CONFIGURATION

N_IMAGES = 50
IMG_SIZE = 64
ROI_SIZE = 32
N_QUBITS = 10
WATERMARK_ANGLE = 0.3 * np.pi
N_SHOTS = 8192
TAU_DEVICE = 0.8627
QWEF_SIGMA = 0.05             
EPS = 1e-8

# Attack parameters
GAUSSIAN_SIGMA = 0.01
BITFLIP_QUBIT = 3
GAN_NODULE_SIZE = 5
GAN_NODULE_INTENSITY = 0.3


# 1. Synthetic medical image generator

def generate_synthetic_medical_image(size=64):
    """Create a grayscale image with smooth background and random blobs."""
    img = np.zeros((size, size), dtype=np.float32)
    x = np.linspace(0, 1, size); y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    img += 0.2 + 0.1 * X + 0.1 * Y
    n_blobs = np.random.randint(2, 5)
    for _ in range(n_blobs):
        cx = np.random.randint(10, size-10)
        cy = np.random.randint(10, size-10)
        rx = np.random.randint(3, 8)
        ry = np.random.randint(3, 8)
        angle = np.random.uniform(0, np.pi)
        blob = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dx = (i - cx) * np.cos(angle) - (j - cy) * np.sin(angle)
                dy = (i - cx) * np.sin(angle) + (j - cy) * np.cos(angle)
                if (dx / rx)**2 + (dy / ry)**2 <= 1:
                    blob[i, j] = 1
        img += blob * np.random.uniform(0.2, 0.5)
    return np.clip(img, 0, 1)


# 2. Quantum encoding & watermarking (FIXED VERSION)

def amplitude_encode(image_2d):
    """Flatten ROI, normalise, pad to 2^10, return state prep circuit and vector."""
    flat = image_2d.flatten()
    norm = np.linalg.norm(flat)
    if norm < 1e-12:
        flat = np.ones_like(flat) / np.sqrt(len(flat))
    else:
        flat = flat / norm
    # Pad to 1024
    target_len = 2**N_QUBITS
    if len(flat) < target_len:
        flat = np.pad(flat, (0, target_len - len(flat)), constant_values=0)
    else:
        flat = flat[:target_len]
    # More precise normalization to avoid Qiskit precision errors
    norm_final = np.linalg.norm(flat)
    if norm_final > 1e-12:
        flat = flat / norm_final
    # Let Qiskit handle final normalization to avoid floating point errors
    init = Initialize(flat, normalize=True)
    qc = QuantumCircuit(N_QUBITS)
    qc.append(init, range(N_QUBITS))
    return qc, flat

def embed_watermark(qc, angle=WATERMARK_ANGLE):
    """Apply phase gates to odd-indexed qubits (watermark pattern)."""
    qc_w = qc.copy()
    qc_w.barrier()
    for i in range(N_QUBITS):
        if i % 2 == 1:
            qc_w.p(angle, i)
    qc_w.barrier()
    return qc_w

def exact_fidelity(qc1, qc2):
    """Exact squared overlap |⟨ψ|φ⟩|² (used in simulation)."""
    sv1 = Statevector.from_instruction(qc1)
    sv2 = Statevector.from_instruction(qc2)
    return np.abs(sv1.evolve(sv2).inner(sv1)) ** 2

def qwef_from_fidelity(F, sigma=QWEF_SIGMA):
    """Quantum Watermark Entanglement Fidelity."""
    if F >= 1.0:
        return 0.0
    return 1.0 - np.exp(-(1.0 - F) / sigma)


# 3. CNN with Grad‑CAM (ResNet18, grayscale adapted)

class ResNetCAM:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = models.resnet18(pretrained=True)
        # Adapt first conv for 1-channel input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.model.conv1.weight[:] = self.model.conv1.weight.mean(dim=1, keepdim=True)
        self.model = self.model.to(device)
        self.model.eval()
        self.target_layer = self.model.layer4[-1]  # last conv in layer4

    def preprocess(self, img_np):
        """Normalise, add batch/channel dims."""
        img_t = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return img_t

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)
        return prob.cpu().numpy()

    def grad_cam(self, x, target_class=None):
        """Generate Grad‑CAM heatmap."""
        x.requires_grad = True
        activations = None
        gradients = None

        def forward_hook(module, inp, out):
            nonlocal activations
            activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_backward_hook(backward_hook)

        output = self.model(x)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        h_fwd.remove()
        h_bwd.remove()

        if gradients is None or activations is None:
            return np.zeros((x.shape[2], x.shape[3]))

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# 4. Classical image similarity

def ssim(img1, img2):
    return structural_similarity(img1, img2, data_range=1.0)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


# 5. Attack functions

def attack_gaussian(roi, sigma=GAUSSIAN_SIGMA):
    return np.clip(roi + np.random.normal(0, sigma, roi.shape), 0, 1)

def attack_bitflip(qc):
    qc_att = qc.copy()
    qc_att.x(BITFLIP_QUBIT)
    return qc_att

def attack_intercept_resend(roi):
    flat = roi.flatten()
    rand = np.random.randn(len(flat))
    rand = rand / np.linalg.norm(rand)
    orig_norm = flat / np.linalg.norm(flat)
    rand = rand - np.dot(rand, orig_norm) * orig_norm
    rnorm = np.linalg.norm(rand)
    if rnorm < 1e-12:
        rand = np.ones_like(rand) / np.sqrt(len(rand))
    else:
        rand = rand / rnorm
    return np.clip(rand.reshape(roi.shape), 0, 1)

def attack_gan_nodule(roi):
    attacked = roi.copy()
    h, w = roi.shape
    cx = np.random.randint(10, h-10)
    cy = np.random.randint(10, w-10)
    for i in range(max(0, cx-GAN_NODULE_SIZE), min(h, cx+GAN_NODULE_SIZE)):
        for j in range(max(0, cy-GAN_NODULE_SIZE), min(w, cy+GAN_NODULE_SIZE)):
            dist = np.sqrt((i-cx)**2 + (j-cy)**2)
            if dist < GAN_NODULE_SIZE:
                attacked[i,j] = min(1.0, attacked[i,j] + GAN_NODULE_INTENSITY * (1 - dist/GAN_NODULE_SIZE))
    return attacked


# 6. Semantic drift

def semantic_drift(cam1, cam2):
    return np.linalg.norm(cam1 - cam2)


# 7. Main evaluation loop

def main():
    print("=" * 80)
    print("QUANTUM-SEMANTIC WATERMARKING: LARGE-SCALE VALIDATION (50 IMAGES)")
    print("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1/7] Initialising ResNet18 on {device}...")
    cam_extractor = ResNetCAM(device)

    # Generate images
    print(f"[2/7] Generating {N_IMAGES} synthetic medical images...")
    images = []
    rois = []
    for i in range(N_IMAGES):
        img = generate_synthetic_medical_image(IMG_SIZE)
        start = (IMG_SIZE - ROI_SIZE) // 2
        roi = img[start:start+ROI_SIZE, start:start+ROI_SIZE]
        images.append(img)
        rois.append(roi)

    # Preprocess a dummy image to warm up
    _ = cam_extractor.preprocess(rois[0])

    # Attack definitions
    attack_list = [
        ('intact', lambda roi, qc: (roi, qc, 'none')),
        ('gaussian', lambda roi, qc: (attack_gaussian(roi), qc, 'image')),
        ('bitflip', lambda roi, qc: (roi, attack_bitflip(qc), 'quantum')),
        ('intercept_resend', lambda roi, qc: (attack_intercept_resend(roi), qc, 'image')),
        ('gan_nodule', lambda roi, qc: (attack_gan_nodule(roi), qc, 'image'))
    ]

    results = []
    pbar = tqdm(total=N_IMAGES * len(attack_list))

    for idx, roi_orig in enumerate(rois):
        # --- Reference ---
        qc_ref, _ = amplitude_encode(roi_orig)
        qc_ref = embed_watermark(qc_ref)

        x = cam_extractor.preprocess(roi_orig)
        prob = cam_extractor.predict_proba(x)
        pred_class = np.argmax(prob)
        cam_ref = cam_extractor.grad_cam(x, target_class=pred_class)

        # --- Attacks ---
        for attack_name, attack_fn in attack_list:
            if attack_name == 'intact':
                roi_att = roi_orig.copy()
                qc_att = qc_ref.copy()
            else:
                roi_att, qc_att, domain = attack_fn(roi_orig, qc_ref)

            # Re‑encode if attack modified image
            if attack_name in ['gaussian', 'intercept_resend', 'gan_nodule']:
                qc_att, _ = amplitude_encode(roi_att)
                qc_att = embed_watermark(qc_att)

            # --- Quantum metrics ---
            F = exact_fidelity(qc_ref, qc_att)
            QWEF = qwef_from_fidelity(F)
            quantum_detected = int(F < TAU_DEVICE)

            # --- Classical image similarity ---
            ss = ssim(roi_orig, roi_att)
            ps = psnr(roi_orig, roi_att)

            # --- Semantic metrics ---
            x_att = cam_extractor.preprocess(roi_att)
            prob_att = cam_extractor.predict_proba(x_att)
            pred_class_att = np.argmax(prob_att)
            cam_att = cam_extractor.grad_cam(x_att, target_class=pred_class_att)
            drift = semantic_drift(cam_ref, cam_att)
            # Simple empirical threshold (will be refined in real use)
            semantic_detected = int(drift > 0.5)
            combined_detected = int(quantum_detected or semantic_detected)

            # Store
            results.append({
                'image_id': idx,
                'attack': attack_name,
                'fidelity': F,
                'qwef': QWEF,
                'quantum_detected': quantum_detected,
                'ssim': ss,
                'psnr': ps,
                'semantic_drift': drift,
                'semantic_detected': semantic_detected,
                'combined_detected': combined_detected,
                'pred_class_orig': pred_class,
                'pred_class_att': pred_class_att,
                'pred_changed': int(pred_class != pred_class_att)
            })
            pbar.update(1)

    pbar.close()
    print("[4/7] Processing complete.")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.to_csv('quantum_semantic_validation_results.csv', index=False)
    print("Results saved ")

    # --- Summary statistics ---
    print("\n[5/7] Summary statistics:")
    print("-" * 80)
    attack_types = df['attack'].unique()
    for attack in attack_types:
        if attack == 'intact':
            continue
        sub = df[df['attack'] == attack]
        print(f"\nAttack: {attack.upper()}")
        print(f"  Fidelity (mean ± std): {sub['fidelity'].mean():.4f} ± {sub['fidelity'].std():.4f}")
        print(f"  QWEF (mean ± std)    : {sub['qwef'].mean():.4f} ± {sub['qwef'].std():.4f}")
        print(f"  Semantic drift       : {sub['semantic_drift'].mean():.4f} ± {sub['semantic_drift'].std():.4f}")
        print(f"  SSIM                 : {sub['ssim'].mean():.4f} ± {sub['ssim'].std():.4f}")
        print(f"  PSNR                 : {sub['psnr'].mean():.4f} ± {sub['psnr'].std():.4f}")
        print(f"  Quantum detection rate : {sub['quantum_detected'].mean():.3f}")
        print(f"  Semantic detection rate: {sub['semantic_detected'].mean():.3f}")
        print(f"  Combined detection rate: {sub['combined_detected'].mean():.3f}")
        print(f"  Prediction changed    : {sub['pred_changed'].mean():.3f}")

    # --- ROC curves ---
    print("\n[6/7] Generating ROC curves...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    detectors = {
        'Quantum Fidelity': 'fidelity',
        'QWEF': 'qwef',
        'Semantic Drift': 'semantic_drift',
        'SSIM': 'ssim'
    }
    for ax_idx, attack in enumerate([a for a in attack_types if a != 'intact']):
        ax = axes[ax_idx]
        mask_attack = (df['attack'] == attack)
        mask_intact = (df['attack'] == 'intact')
        for det_name, col in detectors.items():
            scores = np.concatenate([
                df.loc[mask_intact, col].values,
                df.loc[mask_attack, col].values
            ])
            labels = np.concatenate([
                np.zeros(mask_intact.sum()),
                np.ones(mask_attack.sum())
            ])
            if col in ['ssim', 'psnr']:
                scores = -scores   # lower score → more tampered
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{det_name} (AUC = {roc_auc:.3f})')
        ax.plot([0,1],[0,1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC – {attack.replace("_"," ").title()}')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.svg', format='svg', dpi=300)
    plt.show()
    print("ROC curves saved")

    # --- Scatter plot: Fidelity vs Drift ---
    plt.figure(figsize=(8,6))
    colors = {'intact': 'green', 'gaussian': 'blue', 'bitflip': 'orange',
              'intercept_resend': 'red', 'gan_nodule': 'purple'}
    for attack in attack_types:
        sub = df[df['attack'] == attack]
        plt.scatter(sub['fidelity'], sub['semantic_drift'],
                    label=attack.replace('_',' ').title(),
                    c=colors.get(attack, 'gray'), alpha=0.7, edgecolors='k')
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Semantic threshold (approx)')
    plt.axvline(x=TAU_DEVICE, color='black', linestyle='--', alpha=0.5,
                label=f'Fidelity threshold (τ={TAU_DEVICE})')
    plt.xlabel('Quantum Fidelity')
    plt.ylabel('Semantic Drift ||e - e\'||')
    plt.title('Fidelity vs Semantic Drift – All Attacks')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('fidelity_vs_drift.svg', format='svg', dpi=300)
    plt.show()
    print("Scatter plot saved ")

    print("\n[7/7] Experiment completed ")
    print("=" * 80)

if __name__ == '__main__':
    main()
