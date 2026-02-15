import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import Initialize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

import pandas as pd
from PIL import Image
import os
import kagglehub

N_IMAGES = 100
IMG_SIZE = 64
ROI_SIZE = 32
N_QUBITS = 10
WATERMARK_ANGLE = 0.3 * np.pi
N_SHOTS = 8192
TAU_DEVICE = 0.8627
QWEF_SIGMA = 0.05
EPS = 1e-8

GAUSSIAN_SIGMA = 0.01
BITFLIP_QUBIT = 3
GAN_NODULE_SIZE = 5
GAN_NODULE_INTENSITY = 0.3


def load_chestxray_dataset(data_dir, num_images=100, img_size=64, roi_size=32, use_bbox=True):

    csv_path = os.path.join(data_dir, 'Data_Entry_2017.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist. Please check the path.")

    df = pd.read_csv(csv_path)
    df_sample = df.sample(n=min(num_images, len(df)), random_state=42)

    images = []
    rois = []
    bbox_info = []

    for idx, row in df_sample.iterrows():
        img_path = os.path.join(data_dir, 'images', row['Image Index'])
        try:
            img = Image.open(img_path).convert('L')
        except FileNotFoundError:
            print(f"Warning: image {img_path} not found. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: error opening image {img_path}: {e}")
            continue

        img_resized = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized) / 255.0

        bbox_str = row.get('Bbox [x,y,w,h]', '')
        if use_bbox and pd.notna(bbox_str) and bbox_str != '':
            try:
                x, y, w, h = map(int, bbox_str.split())
                original_size = 1024
                scale = img_size / original_size
                x = int(x * scale)
                y = int(y * scale)
                w = int(w * scale)
                h = int(h * scale)

                x = max(0, min(x, img_size - 1))
                y = max(0, min(y, img_size - 1))
                w = min(w, img_size - x)
                h = min(h, img_size - y)

                if w > 0 and h > 0:
                    roi = img_array[y:y+h, x:x+w]
                    roi_resized = np.array(
                        Image.fromarray((roi*255).astype(np.uint8)).resize((roi_size, roi_size), Image.Resampling.LANCZOS)
                    ) / 255.0
                    rois.append(roi_resized)
                    images.append(img_array)
                    bbox_info.append((x, y, w, h))
                    continue
            except Exception as e:
                print(f"Warning: error processing Bbox for image {row['Image Index']}: {e}")

        start = (img_size - roi_size) // 2
        roi = img_array[start:start+roi_size, start:start+roi_size]
        rois.append(roi)
        images.append(img_array)
        bbox_info.append(None)

    return images, rois, bbox_info


def amplitude_encode(image_2d):
    flat = image_2d.flatten()
    norm = np.linalg.norm(flat)
    if norm < 1e-12:
        flat = np.ones_like(flat) / np.sqrt(len(flat))
    else:
        flat = flat / norm

    target_len = 2**N_QUBITS
    if len(flat) < target_len:
        flat = np.pad(flat, (0, target_len - len(flat)), constant_values=0)
    else:
        flat = flat[:target_len]

    norm_final = np.linalg.norm(flat)
    if norm_final > 1e-12:
        flat = flat / norm_final

    init = Initialize(flat, normalize=True)
    qc = QuantumCircuit(N_QUBITS)
    qc.append(init, range(N_QUBITS))
    return qc, flat


def embed_watermark(qc, angle=WATERMARK_ANGLE):
    qc_w = qc.copy()
    qc_w.barrier()
    for i in range(N_QUBITS):
        if i % 2 == 1:
            qc_w.p(angle, i)
    qc_w.barrier()
    return qc_w


def exact_fidelity(qc1, qc2):
    sv1 = Statevector.from_instruction(qc1)
    sv2 = Statevector.from_instruction(qc2)
    return np.abs(sv1.evolve(sv2).inner(sv1)) ** 2


def qwef_from_fidelity(F, sigma=QWEF_SIGMA):
    if F >= 1.0:
        return 0.0
    return 1.0 - np.exp(-(1.0 - F) / sigma)


class ResNetCAM:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.model.conv1.weight[:] = self.model.conv1.weight.mean(dim=1, keepdim=True)
        self.model = self.model.to(device)
        self.model.eval()
        self.target_layer = self.model.layer4[-1]

    def preprocess(self, img_np):
        img_t = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return img_t

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)
        return prob.cpu().numpy()

    def grad_cam(self, x, target_class=None):
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


def ssim(img1, img2):
    return structural_similarity(img1, img2, data_range=1.0)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


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


def semantic_drift(cam1, cam2):
    return np.linalg.norm(cam1 - cam2)


if __name__ == '__main__':
    main()
