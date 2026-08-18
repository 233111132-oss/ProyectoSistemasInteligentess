/**
 * BioVision ViT - Frontend Controller
 * Maneja subida de archivos, drag & drop, llamada a API Flask, renderizado de resultados y tabs.
 */

let selectedFile = null;
let currentSampleUrl = null;

document.addEventListener('DOMContentLoaded', () => {
    initDropZone();
    initAnalyzeButton();
    initRemoveButton();
});

function initDropZone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Por favor selecciona un archivo de imagen válido (.jpg, .png).');
        return;
    }
    selectedFile = file;
    currentSampleUrl = null;

    const reader = new FileReader();
    reader.onload = (e) => {
        showPreview(e.target.result);
    };
    reader.readAsDataURL(file);
}

function loadSampleImage(url, className) {
    selectedFile = null;
    currentSampleUrl = url;

    showPreview(url);

    // Desplazar suavemente hacia la sección de diagnóstico
    document.getElementById('diagnosis-section').scrollIntoView({ behavior: 'smooth' });

    // Ejecutar diagnóstico automáticamente para experiencia ágil
    setTimeout(() => {
        runInference();
    }, 300);
}

function showPreview(src) {
    const dropPrompt = document.getElementById('dropPrompt');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const btnAnalyze = document.getElementById('btnAnalyze');

    imagePreview.src = src;
    dropPrompt.classList.add('hidden');
    previewContainer.classList.remove('hidden');
    btnAnalyze.disabled = false;
}

function initRemoveButton() {
    const btnRemoveImg = document.getElementById('btnRemoveImg');
    btnRemoveImg.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUploader();
    });
}

function resetUploader() {
    selectedFile = null;
    currentSampleUrl = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('dropPrompt').classList.remove('hidden');
    document.getElementById('previewContainer').classList.add('hidden');
    document.getElementById('btnAnalyze').disabled = true;
    document.getElementById('imagePreview').src = '';
}

function initAnalyzeButton() {
    const btnAnalyze = document.getElementById('btnAnalyze');
    btnAnalyze.addEventListener('click', () => {
        runInference();
    });
}

async function runInference() {
    const loadingState = document.getElementById('loadingState');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const emptyResults = document.getElementById('emptyResults');
    const resultsContent = document.getElementById('resultsContent');

    btnAnalyze.disabled = true;
    loadingState.classList.remove('hidden');

    try {
        let response;
        if (selectedFile) {
            const formData = new FormData();
            formData.append('image', selectedFile);
            response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
        } else if (currentSampleUrl) {
            response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_url: currentSampleUrl })
            });
        } else {
            return;
        }

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            emptyResults.classList.add('hidden');
            resultsContent.classList.remove('hidden');
        } else {
            alert('Error en diagnóstico: ' + (data.error || 'Desconocido'));
        }
    } catch (err) {
        console.error(err);
        alert('Ocurrió un error al procesar la imagen con el modelo ViT.');
    } finally {
        loadingState.classList.add('hidden');
        btnAnalyze.disabled = false;
    }
}

function displayResults(data) {
    const pred = data.prediction;
    const imgs = data.images;

    // Header y banner
    const badge = document.getElementById('classCodeBadge');
    badge.textContent = `CLASE ${pred.class}`;
    badge.style.backgroundColor = pred.color;

    document.getElementById('diseaseName').textContent = pred.name;
    document.getElementById('scientificName').textContent = pred.scientific;
    document.getElementById('confidenceValue').textContent = `${pred.confidence}%`;
    
    const sevBadge = document.getElementById('severityBadge');
    sevBadge.textContent = `Severidad: ${pred.severity}`;

    // Imágenes
    document.getElementById('resOrigImg').src = imgs.original;
    document.getElementById('resAttnImg').src = imgs.attention_overlay;

    // Desglose de Probabilidades
    const probContainer = document.getElementById('probBarsList');
    probContainer.innerHTML = '';
    pred.prob_breakdown.forEach(item => {
        const isMax = item.class === pred.class;
        const div = document.createElement('div');
        div.className = 'prob-item';
        div.innerHTML = `
            <div class="prob-labels">
                <span class="prob-name" style="${isMax ? 'color: #fff; font-weight: 700;' : ''}">
                    Clase ${item.class} - ${item.name}
                </span>
                <span class="prob-val" style="${isMax ? 'color: ' + item.color + ';' : ''}">
                    ${item.percentage}%
                </span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: ${item.percentage}%; background-color: ${item.color};"></div>
            </div>
        `;
        probContainer.appendChild(div);
    });

    // Pestañas agronómicas
    document.getElementById('descText').textContent = pred.description;

    fillList('symptomsList', pred.symptoms);
    fillList('treatmentList', pred.treatment);
    fillList('preventionList', pred.prevention);
}

function fillList(elementId, items) {
    const ul = document.getElementById(elementId);
    ul.innerHTML = '';
    items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        ul.appendChild(li);
    });
}

function switchTab(tabKey) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    const activeBtn = event.currentTarget;
    activeBtn.classList.add('active');
    
    const targetPane = document.getElementById(`pane-${tabKey}`);
    if (targetPane) {
        targetPane.classList.add('active');
    }
}
