"""
Servidor Flask para el Prototipo Interactivo de Clasificación de Enfermedades en Hojas
Utilizando la arquitectura Vision Transformer (ViT)
"""

import sys, os
site_pkg = os.path.abspath('.venv/Lib/site-packages')
if os.path.exists(site_pkg) and site_pkg not in sys.path:
    sys.path.insert(0, site_pkg)

import io
import json
import base64
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torchvision.transforms as transforms

from model.vit_model import VisionTransformer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = ['A', 'B', 'C', 'D']
CLASS_INFO = {
    'A': {
        'code': 'A',
        'name': 'Mancha Foliar / Mancha Bacteriana',
        'scientific': 'Xanthomonas campestris / Septoria spp.',
        'severity': 'Media - Alta',
        'color': '#E53935',
        'badge': 'danger',
        'description': 'Infección bacteriana/fúngica que produce manchas circulares oscuras con bordes amarillentos cloróticos en el tejido foliar. Provoca defoliación prematura y merma en el rendimiento fotosintético.',
        'symptoms': [
            'Pequeñas lesiones circulares o angulares de color marrón o negro.',
            'Halo clorótico amarillento alrededor de las manchas.',
            'Necrosis foliar progresiva que conduce a desecación.'
        ],
        'treatment': [
            'Aplicación de fungicidas a base de cobre (oxicloruro o sulfato de cobre).',
            'Remover y quemar o desechar restos vegetales infectados para evitar dispersión.',
            'Uso de inductores de resistencia (como fosfitos o extractos de gobernadora).'
        ],
        'prevention': [
            'Evitar riego por aspersión que moje las hojas.',
            'Rotación de cultivos y ventilación adecuada entre surcos.',
            'Uso de semillas y plántulas certificadas libres de patógenos.'
        ]
    },
    'B': {
        'code': 'B',
        'name': 'Tizón Foliar / Roya',
        'scientific': 'Phytophthora infestans / Puccinia spp.',
        'severity': 'Muy Alta',
        'color': '#D81B60',
        'badge': 'danger',
        'description': 'Enfermedad fitopatológica altamente destructiva provocada por oomicetos u hongos basidiomicetos. Causa lesiones necróticas extensas de aspecto quemado y pústulas pulverulentas.',
        'symptoms': [
            'Manchas irregulares acuosas que se tornan marrón oscuro a negro rápidamente.',
            'Pústulas o esporulación visible en el envés de la hoja en condiciones de humedad.',
            'Colapso y quemado rápido de hojas enteras y tallos.'
        ],
        'treatment': [
            'Fungicidas sistémicos específicos (Metalaxil, Mefenoxam, Mancozeb).',
            'Eliminación inmediata de plantas con síntomas severos en focos iniciales.',
            'Tratamientos biológicos con Trichoderma harzianum o Bacillus subtilis.'
        ],
        'prevention': [
            'Monitoreo intensivo en temporadas de alta humedad relativa (>85%) y temperaturas moderadas.',
            'Adecuar el drenaje del suelo y evitar encharcamientos.',
            'Densidades de siembra óptimas para garantizar flujo de aire.'
        ]
    },
    'C': {
        'code': 'C',
        'name': 'Moho Foliar',
        'scientific': 'Passalora fulva / Cladosporium fulvum',
        'severity': 'Media',
        'color': '#FB8C00',
        'badge': 'warning',
        'description': 'Patología fúngica común en invernaderos y campos húmedos. Se caracteriza por manchas amarillentas difusas en el haz de la hoja y un fieltro afelpado aterciopelado de color verde oliva o grisáceo en el envés.',
        'symptoms': [
            'Manchas cloróticas pálidas en el haz de la hoja sin bordes definidos.',
            'Moho velloso aterciopelado color marrón olivo en el envés.',
            'Enrollamiento y desecación de las hojas afectadas.'
        ],
        'treatment': [
            'Fungicidas protectores como Clorotalonil o hidróxido de cobre.',
            'Aplicación de biofungicidas a base de aceites esenciales o sales de potasio.',
            'Poda de hojas inferiores para mejorar la ventilación del dosel.'
        ],
        'prevention': [
            'Control estricto de la humedad relativa en invernaderos (mantenerla por debajo del 80%).',
            'Manejo de ventilación cenital y lateral constante.',
            'Selección de cultivares con resistencia genética al moho foliar.'
        ]
    },
    'D': {
        'code': 'D',
        'name': 'Hoja Sana (Sin Enfermedad)',
        'scientific': 'Planta en Estado Óptimo (Healthy Specimen)',
        'severity': 'Ninguna (Óptimo)',
        'color': '#43A047',
        'badge': 'success',
        'description': 'Tejido foliar con desarrollo vigoroso, coloración verde uniforme y sin evidencia de lesiones necróticas, clorosis, pústulas o signos de fitopatógenos. Fotosíntesis en máxima capacidad.',
        'symptoms': [
            'Superficie foliar turgente, limpia y color verde homogéneo.',
            'Nervaduras bien definidas y sin deformaciones.',
            'Ausencia total de manchas cloróticas o pústulas.'
        ],
        'treatment': [
            'No requiere tratamiento curativo.',
            'Mantener el plan de nutrición balanceada (NPK y micronutrientes).',
            'Riego tecnificado por goteo según los requerimientos de la fase fenológica.'
        ],
        'prevention': [
            'Continuar con las buenas prácticas agrícolas (BPA).',
            'Monitoreo preventivo semanal en el cultivo.',
            'Mantenimiento de barreras vivas y control biológico de insectos vectores.'
        ]
    }
}

# Inicializar modelo ViT
model = None
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_or_load_model():
    global model
    if model is None:
        m = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=4,
            embed_dim=192,
            depth=6,
            num_heads=6,
            mlp_ratio=4.0
        ).to(DEVICE)
        
        weights_path = 'model/leaf_vit_model.pt'
        if os.path.exists(weights_path):
            m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            print(f"Pesos de ViT cargados exitosamente desde {weights_path}")
        else:
            print("Aviso: leaf_vit_model.pt no encontrado aún, usando inicialización.")
        m.eval()
        model = m
    return model


def generate_attention_overlay(img_pil, attn_map):
    """
    Genera una imagen en base64 con el mapa de calor superpuesto sobre la hoja.
    """
    img_resized = img_pil.resize((224, 224))
    img_arr = np.array(img_resized) / 255.0

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(img_arr)
    ax.imshow(attn_map, cmap='jet', alpha=0.55)
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/')
def index():
    # Cargar métricas si existen
    metrics_path = 'results/metrics.json'
    metrics_data = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)

    # Obtener una muestra representativa de cada clase para la galería rápida
    sample_gallery = []
    for c in CLASS_NAMES:
        c_dir = os.path.join('Dataset', c)
        if os.path.isdir(c_dir):
            files = [f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                sample_file = files[0]
                sample_gallery.append({
                    'class': c,
                    'title': CLASS_INFO[c]['name'],
                    'filename': sample_file,
                    'path': f"/dataset_image/{c}/{sample_file}"
                })

    return render_template(
        'index.html',
        class_info=CLASS_INFO,
        metrics=metrics_data,
        sample_gallery=sample_gallery
    )


@app.route('/dataset_image/<class_name>/<filename>')
def serve_dataset_image(class_name, filename):
    return send_from_directory(os.path.join('Dataset', class_name), filename)


@app.route('/results_image/<filename>')
def serve_results_image(filename):
    return send_from_directory('results', filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files and 'image_url' not in request.json:
            return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400

        if 'image' in request.files:
            file = request.files['image']
            img_pil = Image.open(file.stream).convert('RGB')
        else:
            rel_path = request.json['image_url'].lstrip('/')
            if rel_path.startswith('dataset_image/'):
                parts = rel_path.split('/')
                img_path = os.path.join('Dataset', parts[1], parts[2])
                img_pil = Image.open(img_path).convert('RGB')
            else:
                return jsonify({'error': 'Ruta de imagen no válida'}), 400

        # Preprocesar imagen
        tensor_img = eval_transform(img_pil).unsqueeze(0).to(DEVICE)

        m = get_or_load_model()
        with torch.no_grad():
            outputs = m(tensor_img)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probabilities))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probabilities[pred_idx]) * 100.0

            # Obtener mapa de autoatención
            attn_map = m.get_attention_map(tensor_img).cpu().numpy()[0]

        # Generar imagen original en base64
        buf_orig = io.BytesIO()
        img_pil.resize((224, 224)).save(buf_orig, format='JPEG')
        buf_orig.seek(0)
        orig_b64 = base64.b64encode(buf_orig.getvalue()).decode('utf-8')

        # Generar overlay de atención
        overlay_b64 = generate_attention_overlay(img_pil, attn_map)

        # Formatear desglose de probabilidades
        prob_breakdown = [
            {
                'class': c,
                'name': CLASS_INFO[c]['name'],
                'percentage': round(float(probabilities[i]) * 100.0, 2),
                'color': CLASS_INFO[c]['color']
            }
            for i, c in enumerate(CLASS_NAMES)
        ]

        response = {
            'success': True,
            'prediction': {
                'class': pred_class,
                'name': CLASS_INFO[pred_class]['name'],
                'scientific': CLASS_INFO[pred_class]['scientific'],
                'confidence': round(confidence, 2),
                'severity': CLASS_INFO[pred_class]['severity'],
                'color': CLASS_INFO[pred_class]['color'],
                'description': CLASS_INFO[pred_class]['description'],
                'symptoms': CLASS_INFO[pred_class]['symptoms'],
                'treatment': CLASS_INFO[pred_class]['treatment'],
                'prevention': CLASS_INFO[pred_class]['prevention'],
                'prob_breakdown': prob_breakdown
            },
            'images': {
                'original': f"data:image/jpeg;base64,{orig_b64}",
                'attention_overlay': f"data:image/png;base64,{overlay_b64}"
            }
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"\n=======================================================")
    print(f" PROTOTIPO VISION TRANSFORMER INICIADO")
    print(f" URL Local: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
