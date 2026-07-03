# Guía Rápida de Comandos (Cheat Sheet)

Esta guía contiene los comandos esenciales para gestionar la infraestructura, el deployment y el entrenamiento del SLM sin GitHub.

## 1. Deployment Local → S3 (Mac)
*Ejecutar desde la carpeta raíz del proyecto (`tfm_slm/`) en tu ordenador local.*

```bash
# Opción 1: Modo TRAINING (descarga → procesa → entrena con GPU)
echo "1" | python3 deploy.py

# Opción 2: Modo INFERENCE (construye imagen, corre chat interactivo en EC2)
echo "2" | python3 deploy.py
```

**Qué hace deploy.py:**
- Archiva código (sin __pycache__, .git, datasets)
- Sube ZIP a S3 bucket `tfm-slm-code`
- Verifica ECR repo existe
- Actualiza `terraform.tfvars` con modo seleccionado

## 2. Gestión de Infraestructura (Terraform)
*Ejecutar desde la carpeta `app/terraform/` en tu ordenador local.*

```bash
# Primera vez solamente (crea S3 backend para estado)
terraform init

# Ver cambios pendientes antes de aplicar
terraform plan

# Desplegar EC2 (descarga código, construye Docker con GPU, ejecuta según modo)
terraform apply -auto-approve -lock=false

# Obtener IP pública de EC2
terraform output ec2_public_ip

# Destruir todo (para ahorrar costes)
terraform destroy -auto-approve -lock=false
```

## 3. Monitorización del Build y Training en EC2
*Ejecutar desde tu Mac (donde tengas tfm-slm.pem).*

```bash
# Monitor en vivo (espera ~5-10 min para Docker build + training)
ssh -i "tfm-slm.pem" ec2-user@<IP> tail -f /var/log/user-data.log

# Si necesitas forzar nuevo build (borra imagen local)
ssh -i "tfm-slm.pem" ec2-user@<IP> docker rmi $(<ID_IMAGEN>) -f
```

## 4. Dentro de EC2 (si necesitas debugging)
*SSH a EC2 y ejecuta:*

```bash
# Ver contenedor activo
docker ps

# Ver logs del contenedor en vivo
docker logs -f <ID_CONTENEDOR>

# Ver estado GPU/VRAM
nvidia-smi
nvidia-smi -l 1  # Actualiza cada segundo

# Ver CloudInit logs si Docker no está levantado aún
sudo tail -f /var/log/cloud-init-output.log
```

## 5. Conexión SSH
*Permisos y conexión básica.*

```bash
# Solo primera vez: permisos correctos para .pem
chmod 400 tfm-slm.pem

# Conectar a instancia
ssh -i "tfm-slm.pem" ec2-user@<IP_PUBLICA>

# Limpiar host key si cambió IP
ssh-keygen -R <IP_PUBLICA>
```

## 6. Deploy y Ejecución de Inferencia (Chat)

### Opción A: Desde Mac (flujo completo)
*Sube código, construye imagen Docker con Flash Attention en EC2, lanza chat interactivo.*

> **Nota:** `docker run -it` requiere terminal real. EC2 UserData construye y sube la imagen a ECR,
> pero el chat hay que lanzarlo manualmente por SSH (sin TTY, el chat se salta automáticamente).

```bash
# PASO 1 — Mac: subir código y provisionar EC2
echo "2" | python3 deploy.py
cd app/terraform && terraform apply -auto-approve -lock=false

# PASO 2 — Esperar build (~10-15 min con Flash Attention)
ssh -i "tfm-slm.pem" ec2-user@<IP>
sudo tail -f /var/log/cloud-init-output.log
# Esperar hasta ver: "Image pushed to ECR"

# PASO 3 — Lanzar chat interactivo (desde dentro de EC2 por SSH)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.eu-south-2.amazonaws.com/tfm-slm:latest"

docker run --gpus all -it \
  -e MODE=inference \
  -v ~/.aws/credentials:/root/.aws/credentials:ro \
  "$IMAGE_URI"
# Descarga checkpoint desde S3 automáticamente y abre sesión "Tú: ..."
# Escribe 'salir' para terminar
```

### Opción B: Si EC2 ya tiene imagen construida (sin rebuild)
*Imagen ya en ECR → solo lanzar contenedor.*

```bash
# En EC2 por SSH:
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.eu-south-2.amazonaws.com/tfm-slm:latest"

# Descargar imagen de ECR si no está local:
aws ecr get-login-password --region eu-south-2 | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.eu-south-2.amazonaws.com"
docker pull "$IMAGE_URI"

# Lanzar chat interactivo:
docker run --gpus all -it \
  -e MODE=inference \
  -v ~/.aws/credentials:/root/.aws/credentials:ro \
  "$IMAGE_URI"
```

**Flujo interno del contenedor en modo inferencia:**
1. Descarga checkpoint desde S3 (`tfm-slm-checkpoints/checkpoint.pt`) si no existe local
2. Carga modelo HybridModel + tokenizer GPT-2
3. Abre sesión de chat interactiva (`Tú: ...`)
4. Escribe `salir` / `exit` / `quit` para terminar

## 7. Benchmarking (Evaluación de Checkpoint)
*Evalúa checkpoint en 300K samples, exporta métricas (loss, perplexity, throughput).*

### Local (Mac)
```bash
# Ejecutar en carpeta raíz del proyecto
uv run tfm-slm-benchmark

# Resultados:
# - Local: output/benchmark_hybrid.json
# - S3: s3://tfm-slm-benchmarks/benchmark_hybrid.json
```

### En EC2
```bash
# Opción 3: Modo BENCHMARKING (descarga dataset + checkpoint, evalúa)
echo "3" | python3 deploy.py

# Provisionar EC2
cd app/terraform && terraform apply -auto-approve -lock=false

# Monitor
ssh -i "tfm-slm.pem" ec2-user@<IP> tail -f /var/log/user-data.log

# EC2 automáticamente:
# 1. Descarga mixed_dataset.tar.gz de S3
# 2. Descarga checkpoint.pt de S3
# 3. Ejecuta tfm-slm-benchmark
# 4. Sube resultados a s3://tfm-slm-benchmarks/benchmark_hybrid.json
```

**Métricas generadas:**
- Loss (promedio)
- Perplexity
- Throughput (tokens/sec)
- Peak memory (GB)

## 8. Pausar y Reanudar (Ahorro de Costes)
*Detiene EC2 pero mantiene checkpoint y imagen ECR.*

```bash
# Pausar (destruye EC2, pero S3 y ECR persisten)
terraform destroy -auto-approve -lock=false

# Verificar que se destruyó
terraform output ec2_public_ip  # Debe estar vacío

# Reanudar (EC2 descarga código, compila si necesario, continúa entrenamiento)
terraform apply -auto-approve -lock=false
```
