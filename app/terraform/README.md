# Infraestructura Terraform para SLM

Este directorio contiene la configuración modular de Terraform para desplegar el entorno de entrenamiento del Small Language Model (SLM).

## Estructura de Archivos

- `provider.tf`: Configuración del proveedor AWS y backend de estado en S3.
- `variables.tf`: Definición de variables (Región, Tipo de Instancia, Key Pair, etc.).
- `ecr.tf`: Repositorio de Amazon ECR para las imágenes Docker.
- `ec2.tf`: Instancia EC2 On-Demand con RTX PRO 6000 Blackwell, Grupos de Seguridad y Roles IAM.
- `outputs.tf`: Valores de salida (IP pública, URL del ECR).

## Requisitos Previos

1. **AWS Credentials (local):**
   - Set `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY` como environment variables o en `~/.aws/credentials`
   - Alternativamente, se puede usar `aws configure` para setup local

2. **Backend S3:**
   - Bucket `tfm-slm-terraform-state` en eu-south-2 para state remoto
   - S3 native locking (no requiere DynamoDB)

3. **SSH Key Pair:**
   - Key Pair llamado `tfm-slm` en eu-south-2 para acceso EC2

## Setup Local

1. **Copiar template de configuración:**
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. **Editar terraform.tfvars con valores locales:**
   ```hcl
   aws_region      = "eu-south-2"
   project_name    = "tfm-slm"
   instance_type   = "g7e.2xlarge"
   ssh_key_name    = "tfm-slm"
   docker_image_tag = "latest"
   spot_price       = "1.50"
   ```

3. **Inicializar Terraform:**
   ```bash
   terraform init
   ```

## Ejecución

```bash
# Ver cambios que se aplicarán
terraform plan

# Aplicar configuración (desplegar infraestructura)
terraform apply

# Destruir infraestructura (eliminar recursos)
terraform destroy
```

Terraform lee automáticamente `terraform.tfvars` en el directorio de trabajo. No necesita versionarse en git.
