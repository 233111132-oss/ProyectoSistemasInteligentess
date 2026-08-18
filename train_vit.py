"""
Script de Entrenamiento y Evaluación Integral para Vision Transformer (ViT)
Clasificación de Enfermedades Foliares en Plantas (ACC, F1, Precision, Recall, Matriz de Confusión)
"""

import sys, os
site_pkg = os.path.abspath('.venv/Lib/site-packages')
if os.path.exists(site_pkg) and site_pkg not in sys.path:
    sys.path.insert(0, site_pkg)

import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

from model.vit_model import VisionTransformer

# Configurar semillas para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['A', 'B', 'C', 'D']
CLASS_DESCRIPTIONS = {
    'A': 'Mancha Foliar / Bacteriana (Early/Bacterial Spot)',
    'B': 'Tizón Foliar / Roya (Late Blight / Rust)',
    'C': 'Moho Foliar (Leaf Mold / Cladosporium)',
    'D': 'Hoja Sana (Healthy Leaf)'
}

class LeafDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, path


def load_dataset_splits(data_dir='Dataset', train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Carga todas las imágenes por clase y realiza una partición estratificada.
    """
    all_files = []
    all_labels = []

    for idx, c in enumerate(CLASS_NAMES):
        c_dir = os.path.join(data_dir, c)
        if not os.path.isdir(c_dir):
            continue
        files = [
            os.path.join(c_dir, f)
            for f in os.listdir(c_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.shuffle(files)
        
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_f = files[:n_train]
        val_f = files[n_train:n_train+n_val]
        test_f = files[n_train+n_val:]
        
        all_files.append((train_f, val_f, test_f))
        print(f"Clase {c} ({CLASS_DESCRIPTIONS[c]}): Total={n} | Train={len(train_f)} | Val={len(val_f)} | Test={len(test_f)}")

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_files, test_labels = [], []

    for idx, (tr, va, te) in enumerate(all_files):
        train_files.extend(tr)
        train_labels.extend([idx] * len(tr))
        
        val_files.extend(va)
        val_labels.extend([idx] * len(va))
        
        test_files.extend(te)
        test_labels.extend([idx] * len(te))

    # Shuffle conjunto de entrenamiento
    train_combined = list(zip(train_files, train_labels))
    random.shuffle(train_combined)
    train_files, train_labels = zip(*train_combined)
    train_files, train_labels = list(train_files), list(train_labels)

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


def train_model(epochs=12, batch_size=32, lr=3e-4):
    print(f"\n=======================================================")
    print(f" INICIANDO EXPERIMENTO: VISION TRANSFORMER (ViT) FOLIARES")
    print(f" Dispositivo de cómputo: {DEVICE}")
    print(f"=======================================================\n")

    os.makedirs('results', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    # 1. Transformaciones de imagen
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Cargar datos
    (tr_f, tr_l), (va_f, va_l), (te_f, te_l) = load_dataset_splits('Dataset')

    train_ds = LeafDataset(tr_f, tr_l, transform=train_transform)
    val_ds = LeafDataset(va_f, va_l, transform=eval_transform)
    test_ds = LeafDataset(te_f, te_l, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Instanciar Vision Transformer
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=4,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ---- FASE DE ENTRENAMIENTO ----
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping para estabilidad en Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # ---- FASE DE VALIDACIÓN ----
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_loss = val_running_loss / total_val
        val_acc = (correct_val / total_val) * 100.0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Época [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train ACC: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val ACC: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model/leaf_vit_model.pt')
            print(f"  >>> Modelo guardado con mejor Val ACC: {val_acc:.2f}%")

    total_training_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_training_time:.2f} segundos.")

    # Cargar los mejores pesos para la evaluación en Test
    model.load_state_dict(torch.load('model/leaf_vit_model.pt'))
    model.eval()

    # ---- EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA (TEST SET) ----
    y_true = []
    y_pred = []
    y_probs = []
    sample_images = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
            if len(sample_images) < 4:
                sample_images.append((paths[0], labels[0].item(), predicted[0].item(), images[0:1]))

    test_acc = accuracy_score(y_true, y_pred) * 100.0
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro_f1 = np.mean(f1) * 100.0
    macro_precision = np.mean(precision) * 100.0
    macro_recall = np.mean(recall) * 100.0

    print(f"\n=======================================================")
    print(f" RESULTADOS FINALES EN TEST SET (CONJUNTO DE PRUEBA)")
    print(f"=======================================================")
    print(f" Accuracy Global (ACC): {test_acc:.2f}%")
    print(f" Macro Precision:       {macro_precision:.2f}%")
    print(f" Macro Recall:          {macro_recall:.2f}%")
    print(f" Macro F1-Score:        {macro_f1:.2f}%")
    print("-------------------------------------------------------")
    print(" Reporte de Clasificación por Clase:")
    print(classification_report(y_true, y_pred, target_names=[f"{c} - {CLASS_DESCRIPTIONS[c]}" for c in CLASS_NAMES], digits=4))

    # Guardar métricas en JSON
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    metrics_dict = {
        'test_accuracy': float(test_acc),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'classes': CLASS_NAMES,
        'class_descriptions': CLASS_DESCRIPTIONS,
        'per_class': {
            c: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            for i, c in enumerate(CLASS_NAMES)
        },
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist(),
        'history': history,
        'training_time_seconds': total_training_time
    }

    with open('results/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

    # 4. Generar Gráficas de Resultados en Alta Definición (300 DPI)
    generate_result_plots(history, cm, cm_norm, precision, recall, f1, model, sample_images)

    return metrics_dict


def generate_result_plots(history, cm, cm_norm, precision, recall, f1, model, sample_images):
    """
    Genera todas las gráficas para el Reporte de Práctica y la Presentación.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # --- 1. Curvas de Aprendizaje (Loss y ACC vs Épocas) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    epochs_range = range(1, len(history['train_loss']) + 1)

    # Gráfica de Pérdida (Cross-Entropy Loss)
    ax1.plot(epochs_range, history['train_loss'], 'o-', color='#1E88E5', linewidth=2.2, label='Pérdida Entrenamiento')
    ax1.plot(epochs_range, history['val_loss'], 's--', color='#E53935', linewidth=2.2, label='Pérdida Validación')
    ax1.set_title('Convergencia de Función de Pérdida (Loss)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Épocas de Entrenamiento', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Gráfica de Precisión (Accuracy - ACC)
    ax2.plot(epochs_range, history['train_acc'], 'o-', color='#00897B', linewidth=2.2, label='ACC Entrenamiento (%)')
    ax2.plot(epochs_range, history['val_acc'], 's--', color='#FB8C00', linewidth=2.2, label='ACC Validación (%)')
    ax2.set_title('Evolución de Precisión (Accuracy - ACC)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Épocas de Entrenamiento', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('results/loss_accuracy_curves.png', dpi=300)
    plt.close()
    print(" Gráfica guardada: results/loss_accuracy_curves.png")

    # --- 2. Matriz de Confusión (Conteos y Porcentajes Normalizados) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Matriz Absoluta
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax1,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    ax1.set_title('Matriz de Confusión (Muestras de Prueba)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Clase Predicha por ViT', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Clase Real (Ground Truth)', fontsize=11, fontweight='bold')

    # Matriz Normalizada (%)
    sns.heatmap(
        cm_norm * 100.0, annot=True, fmt='.1f', cmap='Greens', cbar=True, ax=ax2,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    ax2.set_title('Matriz de Confusión Normalizada (%)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Clase Predicha por ViT', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Clase Real (Ground Truth)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300)
    plt.close()
    print(" Gráfica guardada: results/confusion_matrix.png")

    # --- 3. Métricas por Clase (Precision, Recall, F1) ---
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    rects1 = ax.bar(x - width, precision * 100.0, width, label='Precisión (%)', color='#2b5c8f', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x, recall * 100.0, width, label='Sensibilidad / Recall (%)', color='#3da4ab', edgecolor='black', linewidth=0.8)
    rects3 = ax.bar(x + width, f1 * 100.0, width, label='F1-Score (%)', color='#f6cd61', edgecolor='black', linewidth=0.8)

    ax.set_title('Desempeño de Clasificación por Categoría Foliar (ViT)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Clase / Condición Foliar', fontsize=11, fontweight='bold')
    ax.set_ylabel('Porcentaje (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Clase {c}\n({CLASS_DESCRIPTIONS[c].split('(')[0].strip()})" for c in CLASS_NAMES], fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Etiquetas numéricas sobre las barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig('results/per_class_metrics.png', dpi=300)
    plt.close()
    print(" Gráfica guardada: results/per_class_metrics.png")

    # --- 4. Mapa de Atención Visual (Explainable AI / Attention Rollout) ---
    if sample_images:
        path, true_lbl, pred_lbl, img_tensor = sample_images[0]
        img_tensor = img_tensor.to(DEVICE)
        attn_map = model.get_attention_map(img_tensor).cpu().numpy()[0]

        raw_img = Image.open(path).convert('RGB').resize((224, 224))
        raw_arr = np.array(raw_img)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

        # Imagen original
        ax1.imshow(raw_arr)
        ax1.set_title(f"Hoja Original\nReal: Clase {CLASS_NAMES[true_lbl]}", fontsize=11, fontweight='bold')
        ax1.axis('off')

        # Mapa de autoatención ViT
        im2 = ax2.imshow(attn_map, cmap='jet')
        ax2.set_title("Mapa de Autoatención ViT\n(Multi-Head Self-Attention)", fontsize=11, fontweight='bold')
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Superposición (Overlay)
        ax3.imshow(raw_arr)
        ax3.imshow(attn_map, cmap='jet', alpha=0.55)
        ax3.set_title(f"Interpretabilidad Fitosanitaria\nPredicción: Clase {CLASS_NAMES[pred_lbl]}", fontsize=11, fontweight='bold')
        ax3.axis('off')

        plt.tight_layout()
        plt.savefig('results/attention_sample.png', dpi=300)
        plt.close()
        print(" Gráfica guardada: results/attention_sample.png")

if __name__ == '__main__':
    train_model(epochs=12, batch_size=32, lr=3e-4)
