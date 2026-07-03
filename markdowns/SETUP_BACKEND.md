# Setup Backend: S3 para Código, Checkpoints y Estado de Terraform

Ejecuta estos comandos **una sola vez** en tu terminal local (Mac) para crear los recursos necesarios:
- Bucket para código (deploy.py sube aquí)
- Bucket para checkpoints (training guarda aquí)
- Bucket para estado de Terraform (terraform.tfstate)

## Prerequisitos

```bash
# Verificar que tienes AWS CLI y credenciales configuradas
aws sts get-caller-identity
# Debe mostrar tu Account ID, User ARN y Account ID
```

## 1. Crear Buckets S3 (Ejecutar Una Sola Vez)

### 1a. Bucket para Código (`tfm-slm-code`)
Deploy.py archiva y sube el código aquí. EC2 lo descarga al iniciar.

```bash
aws s3api create-bucket \
    --bucket tfm-slm-code \
    --region eu-south-2 \
    --create-bucket-configuration LocationConstraint=eu-south-2
```

### 1b. Bucket para Checkpoints (`tfm-slm-checkpoints`)
Training guarda automáticamente checkpoints aquí después de cada época. Chat lo descarga si no está local.

```bash
aws s3api create-bucket \
    --bucket tfm-slm-checkpoints \
    --region eu-south-2 \
    --create-bucket-configuration LocationConstraint=eu-south-2
```

### 1c. Bucket para Datasets Procesados (`tfm-slm-datasets`)
Dataset procesado (.arrow) se sube aquí tras primer procesado. Runs futuros lo restauran directamente — evita reprocesar ~700K samples.

```bash
aws s3api create-bucket \
    --bucket tfm-slm-datasets \
    --region eu-south-2 \
    --create-bucket-configuration LocationConstraint=eu-south-2
```

### 1d. Bucket para Benchmarks (`tfm-slm-benchmarks`)
Almacena resultados de benchmarking (métricas, perplexity, throughput) en JSON. Script `tfm-slm-benchmark` sube automáticamente.

```bash
aws s3api create-bucket \
    --bucket tfm-slm-benchmarks \
    --region eu-south-2 \
    --create-bucket-configuration LocationConstraint=eu-south-2
```

### 1e. Bucket para Estado Terraform (`tfm-slm-terraform-state`)
Almacena `terraform.tfstate` de forma centralizada (permite múltiples ejecuciones seguras).

```bash
aws s3api create-bucket \
    --bucket tfm-slm-terraform-state \
    --region eu-south-2 \
    --create-bucket-configuration LocationConstraint=eu-south-2
```

## 2. Configurar Versionado (Backup de Estado)

Habilita versionado en el bucket de estado para recuperar versiones anteriores si algo falla.

```bash
aws s3api put-bucket-versioning \
    --bucket tfm-slm-terraform-state \
    --versioning-configuration Status=Enabled
```

## 3. Validar Buckets Creados

```bash
# Listar todos tus buckets
aws s3 ls

# Verificar todos los buckets están en eu-south-2
aws s3api list-buckets --query 'Buckets[?contains(Name, `tfm-slm`)]'

# Verificar versionado está habilitado en terraform-state
aws s3api get-bucket-versioning --bucket tfm-slm-terraform-state
```

---

## Flujo Completo (Después de Setup)

```bash
# 1. Preparar código y subirlo a S3 (elige modo: 1=train, 2=inference)
echo "1" | python3 deploy.py

# 2. Provisionar EC2 desde Terraform (descarga código, construye Docker con GPU)
cd app/terraform
terraform init      # Una sola vez
terraform apply -auto-approve -lock=false

# 3. Monitorizar build y training
ssh -i "tfm-slm.pem" ec2-user@<IP> tail -f /var/log/user-data.log
```

---

## Notas

- **S3 Native Locking**: Terraform usa `use_lockfile = true` (S3 native locking, no necesita DynamoDB).
- **Credenciales AWS**: Se leen automáticamente desde `~/.aws/credentials` (resultado de `aws configure`).
- **Región**: Todos los buckets en `eu-south-2` (editar en comandos si prefieres otra región).
- **Buckets únicos**: Los nombres de buckets S3 son **globales**. Si ya existen, usa nombres diferentes en `deploy.py`, `terraform.tfvars` y estos comandos.
