# tfm-slm: Modelo de Lenguaje Pequeño con Arquitectura Híbrida Transformer-GRU

tfm-slm es un modelo de lenguaje de escala reducida (Small Language Model) desarrollado en el marco de un Trabajo de Fin de Máster (TFM). El proyecto implementa una arquitectura híbrida personalizada que integra mecanismos de atención global (Transformers) con la eficiencia del refinamiento secuencial (GRU), optimizando el rendimiento para hardware NVIDIA de última generación y flujos de trabajo MLOps profesionales.

## Características Principales

*   Arquitectura Híbrida Integrada: 12 capas compuestas por bloques híbridos (HybridBlock) que fusionan Multi-Head Attention y GRU.
*   Eficiencia en Parámetros: Configuración de aproximadamente 167M de parámetros con implementación de Weight Tying entre embeddings y la cabeza de salida.
*   Entrenamiento de Alto Rendimiento: Soporte nativo para precisión bfloat16 y aceleración de matmuls mediante TF32, optimizado para NVIDIA RTX PRO 6000 (Arquitectura Blackwell).
*   Pipeline de Datos Profesional: Servicios desacoplados siguiendo principios SOLID, utilizando el formato Apache Arrow para una gestión eficiente de la memoria mediante memory-mapping.
*   MLOps Determinista: Entornos reproducibles mediante el gestor uv, construcción de contenedores optimizada con caché global y CI/CD dirigido por Pull Requests.

## Arquitectura del Modelo: HybridBlock

A diferencia de las arquitecturas híbridas convencionales que se limitan a concatenar capas de distinto tipo, tfm-slm propone una integración profunda en cada nivel del modelo a través del componente HybridBlock. Cada uno de los 12 bloques del sistema ejecuta la siguiente secuencia de procesamiento optimizada:

1.  Normalización Previa (Pre-LayerNorm): Aplicada antes de cada sub-componente para garantizar la estabilidad numérica durante el entrenamiento de la arquitectura desde cero.
2.  Unidad Recurrente Puerta (GRU): Posicionada primero para capturar patrones locales y refinamiento secuencial. Mantiene estados ocultos persistentes a través de las 12 capas, permitiendo verdadera recurrencia sin coste cuadrático.
3.  Auto-Atención Multi-Cabeza (MHA): Configurada con 12 cabezas de atención, se encarga de capturar dependencias globales y relaciones semánticas de largo alcance en secuencias de hasta 1024 tokens, refinando los patrones locales del GRU.
4.  Red de Realimentación (FFN): Una estructura MLP con factor de expansión 4 y activación GELU para el procesamiento de características de alto nivel.

El diseño se completa con una estrategia de Weight Tying, que reduce el uso de memoria de video (VRAM) en un 18% al compartir pesos entre la capa de entrada y salida, permitiendo batch sizes más elevados durante el entrenamiento.

## Mejoras Implementadas (Mayo 2026)

Se han aplicado **10 mejoras críticas** a la arquitectura híbrida para mejorar estabilidad, visibilidad, rendimiento e intensidad computacional:

### Mejoras en `app/model/architecture.py`
1. **Persistent GRU Hidden States**: Los estados ocultos del GRU se mantienen a través de todas las 12 capas, permitiendo verdadera recurrencia entre bloques.
2. **GRU Pre-Attention**: Reposicionamiento de componentes — GRU procesa primero (captura patrones locales), luego Attention (refina con contexto global).
3. **Inicialización Ortogonal RNN**: Pesos `weight_ih` y `weight_hh` inicializados ortogonalmente para estabilidad de gradientes.

### Mejoras en `app/training/trainer.py`
4. **Gradient Clipping**: Control automático de gradientes para evitar explosiones típicas de RNNs. Parámetro configurable `grad_clip_norm=1.0`.
5. **Component Contribution Logging**: Logueo cada N steps de normas GRU/Attention/MLP y ratio GRU/Attn para validar balance híbrido.
6. **Validation Loop**: Evaluación periódica en validation set para detectar overfitting temprano.

### Optimizaciones de Rendimiento
7. **Flash Attention** (`app/model/architecture.py`): 
   - 2-3x speedup en GPUs NVIDIA modernas. Reduce memoria O(n²) → O(n).
   - Import opcional (`try/except ImportError`): si `flash_attn` no está instalado, cae automáticamente a `F.scaled_dot_product_attention` con máscara causal.
   - Instalado en el `Dockerfile` desde wheel pre-compilado (x86_64/Linux); en Mac no está disponible y se usa el fallback.
8. **Learning Rate Scheduler** (`app/training/trainer.py`): Cosine annealing con warmup mejora convergencia 15% en épocas y +1-2% en accuracy.

### Herramientas Nuevas
9. **HybridArchitectureAnalyzer** (`app/utils/analyzer.py`): Herramienta de análisis detallado con métodos para analizar componentes, registrar métricas y exportar estadísticas a JSON.
10. **Test Suite** (`tests/test_architecture.py`): 7 tests automatizados validando forward pass, persistent states, gradient flow, weight initialization, y análisis de componentes.

### Validación
```bash
# Ejecutar tests de arquitectura
uv run pytest tests/test_architecture.py -v
# Resultado: 7/7 tests pasando ✅
```

### Uso en Training
```python
from app.training.trainer import TrainingService

service = TrainingService()
service.train(
    epochs=3,
    grad_clip_norm=1.0,      # Control de gradientes
    log_metrics_every=100,   # Loguear cada 100 steps
    validate_every=500,      # Validar cada 500 steps
)
# Genera metrics.json junto a checkpoint automáticamente
```

## Estrategia de Mezcla de Datos (Data Mixing)

El modelo utiliza una combinación estratégica de cuatro fuentes de datos abiertos, ponderadas como sigue:
*   OpenAssistant (30%) y UltraChat (30%): naturalidad en el diálogo y consistencia en contextos largos.
*   Alpaca (20%) y ShareGPT (20%): capacidades resolutivas y seguimiento de instrucciones.

UltraChat actúa como fuente elástica: al tener más de 1.4M de filas disponibles, absorbe el déficit que dejan las fuentes de tamaño fijo (p. ej. Alpaca solo tiene 52,002 filas en total) para que el total siga alcanzando `total_samples`.

### Split del Dataset

El dataset combinado (494,876 muestras) se divide en dos splits disjuntos:
*   **346,439 muestras** (`train`): entrenamiento.
*   **148,437 muestras** (`benchmark`): holdout fijo, no visto durante el entrenamiento, usado para benchmarking.

## Instalación y Uso

### Requisitos Previos
*   uv (Gestor de paquetes de Python)
*   Docker (Opcional, para ejecución en contenedores)
*   GPU NVIDIA con 96GB de VRAM (Optimizado para: NVIDIA RTX PRO 6000)

### Configuración Local
```bash
# Sincronizar el entorno e instalar dependencias
uv sync

# Ejecutar el pipeline completo (Descarga -> Procesamiento -> Entrenamiento)
uv run tfm-slm
```

### Docker
```bash
# Construcción de la imagen optimizada
docker build -t tfm-slm:latest .

# Ejecución del contenedor con soporte de GPU
docker run --rm --gpus all tfm-slm:latest
```

### Deployment a AWS (sin GitHub)

Pipeline local → S3 → EC2 con GPU (buildea + entrena).

**Requisitos Previos:**
- AWS credentials en `~/.aws/credentials` (de `aws configure`)
- Terraform instalado
- SSH key `tfm-slm.pem` local

**Flujo Completo:**

1. Configurar Terraform (una sola vez):
```bash
cp app/terraform/terraform.tfvars.example app/terraform/terraform.tfvars
# Editar terraform.tfvars: región, instance_type (g7e.2xlarge), ssh_key_name, etc.
```

2. Deploy código a S3 (selecciona modo: train, inference o benchmark):
```bash
echo "1" | python3 deploy.py    # 1=train, 2=inference, 3=benchmark
# → Archiva código
# → Sube ZIP a S3 bucket (tfm-slm-code)
# → Verifica ECR repo
# → Guarda deployment_mode en terraform.tfvars
```

3. Crea infraestructura EC2 con GPU:
```bash
cd app/terraform
terraform init          # Una sola vez
terraform apply -auto-approve -lock=false
# → Crea EC2 g7e.2xlarge (RTX PRO 6000)
# → EC2 UserData ejecuta build-and-train.sh automáticamente
```

4. Monitor entrenamiento (en otra terminal):
```bash
ssh -i "tfm-slm.pem" ec2-user@<IP> tail -f /var/log/user-data.log
# <IP> aparece en terraform output
```

**EC2 ejecuta automáticamente:**
- Descarga código de S3 → `/home/ec2-user/tfm-slm`
- **Modo Train**: buildea Docker (flash-attn compila con GPU) → pushea ECR → entrena (15 epochs ~2-4h)
- **Modo Inference**: descarga imagen ECR → ejecuta chat interactivo
- **Modo Benchmark**: descarga dataset (split `benchmark`) + checkpoint de S3 → ejecuta `tfm-slm-benchmark` → sube resultados a S3

**Deploy.py archiva:**
- `pyproject.toml`, `uv.lock`, `Dockerfile`, `entrypoint.sh`, `build-and-train.sh`, `app/` (sin __pycache__)
- Excluye: `.git`, datasets, tests, documentación

**Nota:** AWS credentials leídas automáticamente de `~/.aws/credentials`.

### Deployment a Kubernetes local (ArgoCD/Kind, sin GitHub)

Alternativa al flujo EC2 anterior: despliega el chat como servicio en un clúster de Kubernetes local (Kind), gestionado vía Kustomize o simulando GitOps con ArgoCD. Ver `argocd-cluster/README.md` para la guía completa paso a paso.

```bash
# Build de la imagen y creación del clúster local
docker build -t tfm-slm:latest .
kind create cluster --name tfm-cluster
kind load docker-image tfm-slm:latest --name tfm-cluster
kubectl create namespace tfm-slm

# Despliegue directo con Kustomize (sin ArgoCD)
kubectl apply -k argocd-cluster/manifests/

# Copiar checkpoint local al pod y exponer el servicio
kubectl cp .output/checkpoint.pt tfm-slm/<POD_NAME>:/app/data/checkpoint.pt
kubectl rollout restart deployment/tfm-slm-chat -n tfm-slm
kubectl port-forward svc/tfm-slm-chat-service -n tfm-slm 8000:8000
```

Expone una interfaz web en `http://localhost:8000` y el endpoint `POST /api/chat`.

## Estructura del Proyecto

*   app/: Código fuente principal en Python.
    *   model/: Implementación técnica de la arquitectura híbrida mejorada.
    *   training/: Servicio de entrenamiento optimizado para NVIDIA RTX PRO 6000 con gradient clipping y validación.
    *   dataset/: Servicios de adquisición, armonización y mezcla de datos.
    *   utils/: Herramientas de análisis (HybridArchitectureAnalyzer).
    *   chat/: Interfaz de chat interactivo post-entrenamiento (API REST).
    *   terraform/: Configuración IaC para AWS (EC2, S3, ECR).
    *   inference.py: Entry point de inferencia — carga checkpoint desde S3 y lanza sesión de chat.
    *   ask_chat.py: Cliente HTTP que lanza 10 preguntas de prueba al API del chat (requiere `kubectl port-forward`).
    *   benchmarking.py: Evaluación sobre el split `benchmark` (148,437 muestras) — exporta métricas a `.output/benchmark_hybrid.json`.
    *   benchmark_lmeval.py: Benchmarks zero-shot de opción múltiple (HellaSwag, ARC-Easy, ARC-Challenge, PIQA) vía log-likelihood scoring, estilo lm-evaluation-harness.
*   argocd-cluster/: Manifiestos de Kubernetes (Kustomize) y ArgoCD para desplegar el chat en un clúster local (Kind). Ver `argocd-cluster/README.md`.
*   tests/: Suite de tests (main, architecture, dataset).
*   deploy.py: Script local para upload código a S3 y build Docker en ECR.

## Testing y Validación

Tests compartimentalizados:
*   `tests/test_main.py`: Tests de orquestación de servicios.
*   `tests/test_architecture.py`: 7 tests de arquitectura híbrida (hidden states, gradient flow, componentes).
*   `tests/dataset/test_dataset_downloader.py`: Tests del pipeline de descarga de datasets.

Ejecutar todo:
```bash
uv run pytest tests/ -v
```

## Benchmarking

Evaluación sobre el split `benchmark` (148,437 muestras), holdout fijo disjunto del `train` y no visto durante el entrenamiento, en NVIDIA RTX PRO 6000:

```bash
uv run python app/benchmarking.py
# Genera .output/benchmark_hybrid.json y sube a S3
```

### Resultados (`benchmark_hybrid.json`)

| Métrica | Valor |
|---|---|
| Loss (Cross-Entropy) | 1.6008 |
| Perplexity | 4.96 |
| Token Accuracy (Top-1) | 68.25% |
| Top-5 Accuracy | 82.65% |
| Top-10 Accuracy | 86.82% |
| Latencia por token | 7.00 ms |
| Throughput (batch) | 90,267 tokens/s |
| Throughput (single token) | 143 tokens/s |
| Parámetros totales | 166,980,864 (~167M) |
| Tamaño del modelo | 636.98 MB |
| Memoria pico (VRAM) | 66.76 GB |
| Tiempo total evaluación | 1,682.24 s (~28 min) |

### Prueba del API de Chat

Con el servicio desplegado en Kubernetes, `ask_chat.py` lanza 10 preguntas de prueba al endpoint:

```bash
# Exponer el servicio localmente
kubectl port-forward svc/tfm-slm-chat-service -n tfm-slm 8000:8000

# En otra terminal, lanzar las preguntas de prueba
uv run python app/ask_chat.py
# También acepta variable de entorno:
# CHAT_URL=http://localhost:8000 uv run python app/ask_chat.py
```
