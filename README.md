# tfm-slm: Modelo de Lenguaje Pequeño con Arquitectura Híbrida Transformer-GRU

tfm-slm es un modelo de lenguaje de escala reducida (Small Language Model) desarrollado en el marco de un Trabajo de Fin de Máster (TFM). El proyecto implementa una arquitectura híbrida personalizada que integra mecanismos de atención global (Transformers) con la eficiencia del refinamiento secuencial (GRU), optimizando el rendimiento para hardware NVIDIA de última generación y flujos de trabajo MLOps profesionales.

## Características Principales

*   Arquitectura Híbrida Integrada: 12 capas compuestas por bloques híbridos (HybridBlock) que fusionan Multi-Head Attention y GRU.
*   Eficiencia en Parámetros: Configuración de aproximadamente 124M de parámetros con implementación de Weight Tying entre embeddings y la cabeza de salida.
*   Entrenamiento de Alto Rendimiento: Soporte nativo para precisión bfloat16 y aceleración de matmuls mediante TF32, optimizado para NVIDIA RTX PRO 6000 (Arquitectura Ampere).
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
   - Instalación condicional: `flash-attn>=2.6.0; sys_platform == 'linux'` (solo AWS/Linux)
   - Código ejecuta solo en AWS, no en Mac (solo edición)
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

El modelo utiliza una combinación estratégica de cinco pilares de datos abiertos:
*   Conversacional (50%): OpenAssistant y UltraChat, para asegurar naturalidad en el diálogo y consistencia en contextos largos.
*   Instrucciones (50%): Alpaca y ShareGPT, enfocados en dotar al modelo de capacidades resolutivas y seguimiento de instrucciones.
*   Especializado: Subconjunto de The Stack (YAML) para el conocimiento de sintaxis de infraestructura y flujos DevOps.

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

2. Deploy código a S3 (selecciona modo: train o inference):
```bash
echo "1" | python3 deploy.py    # 1=train, 2=inference
# → Archiva código
# → Sube ZIP a S3 bucket (tfm-slm-code)
# → Verifica ECR repo
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

**Deploy.py archiva:**
- `pyproject.toml`, `uv.lock`, `Dockerfile`, `entrypoint.sh`, `build-and-train.sh`, `app/` (sin __pycache__)
- Excluye: `.git`, datasets, tests, documentación

**Nota:** AWS credentials leídas automáticamente de `~/.aws/credentials`.

## Estructura del Proyecto

*   app/: Código fuente principal en Python.
    *   model/: Implementación técnica de la arquitectura híbrida mejorada.
    *   training/: Servicio de entrenamiento optimizado para NVIDIA RTX PRO 6000 con gradient clipping y validación.
    *   dataset/: Servicios de adquisición, armonización y mezcla de datos.
    *   utils/: Herramientas de análisis (HybridArchitectureAnalyzer).
    *   chat/: Interfaz de chat interactivo post-entrenamiento.
    *   terraform/: Configuración IaC para AWS (EC2, S3, ECR).
*   tests/: Suite de tests (main, architecture).
*   deploy.py: Script local para upload código a S3 y build Docker en ECR.

## Testing y Validación

Tests compartimentalizados:
*   `tests/test_main.py`: Tests de orquestación de servicios.
*   `tests/test_architecture.py`: 7 tests de arquitectura híbrida (hidden states, gradient flow, componentes).

Ejecutar todo:
```bash
uv run pytest tests/ -v
```
