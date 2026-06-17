# Setup de Kubernetes y ArgoCD en Local

Este directorio contiene las configuraciones necesarias para desplegar, entrenar e interactuar con el modelo tfm-slm en un clúster de Kubernetes local utilizando ArgoCD para la gestión GitOps o directamente a través de Kustomize.

---

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado en tu máquina local:
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/) o similar (Rancher Desktop, Podman).
*   [kubectl](https://kubernetes.io/docs/tasks/tools/) (Línea de comandos de Kubernetes).
*   [Kind](https://kind.sigs.k8s.io/) (Herramienta para crear clústeres locales con contenedores Docker).
*   [Argo CD CLI](https://argo-cd.readthedocs.io/en/stable/cli_installation/) (Opcional, pero recomendado).

---

## Estructura de Archivos

*   `application.yaml`: Manifiesto de la aplicación ArgoCD que apunta a este repositorio.
*   `manifests/`: Manifiestos base de Kubernetes agrupados mediante Kustomize:
    *   `kustomization.yaml`: Orquestador de recursos.
    *   `pvc.yaml`: Volumen persistente local para datos.
    *   `configmap.yaml`: Variables de entorno de la aplicación (apunta a `/app/data/checkpoint.pt`).
    *   `secrets.yaml`: Plantilla para credenciales de AWS (S3).
    *   `training-job.yaml`: Job de Kubernetes para ejecutar la fase de entrenamiento.
    *   `chat-deployment.yaml`: Deployment para el servidor API de inferencia.
    *   `service.yaml`: Servicio de Kubernetes para exponer el chatbot.

---

## Carga del Checkpoint Local en Kubernetes

Dado que el clúster local de Kind corre dentro de un contenedor Docker aislado y no tiene acceso directo a los archivos de tu Mac por defecto, se utiliza el volumen persistente del clúster (`PersistentVolumeClaim`) en lugar de `hostPath` para mayor portabilidad y estabilidad.

Una vez que el pod esté desplegado, se puede copiar el checkpoint directamente desde el disco del Mac hacia el almacenamiento persistente del pod.

---

## Paso a Paso: Configuración del Clúster

### 1. Construir la Imagen de Docker del Modelo
Construye la imagen Docker localmente con la etiqueta `tfm-slm:latest`:
```bash
docker build -t tfm-slm:latest .
```

---

### 2. Levantar el Clúster de Kubernetes Local con Kind
Crea el clúster local e inyecta la imagen construida directamente en los nodos:
```bash
# Crear clúster
kind create cluster --name tfm-cluster

# Cargar imagen local en el clúster
kind load docker-image tfm-slm:latest --name tfm-cluster
```

---

### 3. Crear el Namespace del Proyecto
Crea el espacio de nombres para aislar los recursos de la aplicación:
```bash
kubectl create namespace tfm-slm
```

---

## Método de Despliegue

### Opción A: Despliegue Local Rápido (Recomendado para pruebas sin Git)
Si no quieres usar Git u organizar un flujo GitOps en este momento, puedes aplicar directamente la carpeta de manifiestos usando Kustomize:

```bash
kubectl apply -k manifests/
```

---

### Opción B: GitOps Simulado con ArgoCD en Local (Sin GitHub)
Si no deseas publicar tu código en un repositorio público/privado de GitHub pero quieres simular el flujo GitOps, puedes arrancar un servidor Git local en tu propio Mac y hacer que ArgoCD lea de él:

1. **Iniciar el Git Daemon en tu Mac:**
   Abre una terminal nueva en tu Mac y arranca el servidor Git integrado:
   ```bash
   git daemon --base-path=/Users/guille/Documents/GITHUB --export-all --reuseaddr --verbose
   ```
   *Nota: Mantén esta terminal abierta. Esto permite que el clúster acceda a tu carpeta local usando `git://host.docker.internal/tfm-slm`.*

2. **Confirmar los cambios localmente en Git:**
   Asegúrate de hacer un commit local para que el servidor Git detecte las últimas modificaciones del código:
   ```bash
   git add .
   git commit -m "Configure local GitOps sync"
   ```

3. **Instalar Argo CD en el Clúster:**
   ```bash
   # Crear namespace de ArgoCD
   kubectl create namespace argocd

   # Instalar mediante server-side apply
   kubectl apply --server-side -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
   ```

4. **Esperar a que ArgoCD inicie:**
   Monitorea el estado hasta que el pod de `argocd-server` cambie a `Running`:
   ```bash
   kubectl get pods -n argocd -w
   ```

5. **Obtener la Contraseña de Administrador:**
   ```bash
   kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo
   ```

6. **Desplegar la Aplicación en ArgoCD:**
   Aplica el manifiesto de la aplicación (que ya viene preconfigurado con `git://host.docker.internal/tfm-slm`):
   ```bash
   kubectl apply -f argocd-cluster/application.yaml
   ```

7. **Exponer el Servidor de ArgoCD en Localhost:**
   ```bash
   kubectl port-forward svc/argocd-server -n argocd 8080:443
   ```
   *Entra en tu navegador a https://localhost:8080 con el usuario `admin` e inicia sesión para ver y sincronizar tu aplicación.*

---

## Verificación, Carga del Checkpoint e Interacción con el Chatbot

### 1. Verificar el estado de los recursos
```bash
kubectl get all -n tfm-slm
```

### 2. Copiar tu Checkpoint local al clúster (Paso requerido en local)
Dado que el pod utilizará el volumen persistente, debes copiar tu archivo `checkpoint.pt` dentro del volumen del pod:

1. Obtén el nombre exacto del pod del chat:
   ```bash
   kubectl get pods -n tfm-slm -l app=tfm-slm-chat
   ```
2. Copia tu checkpoint local dentro del pod:
   ```bash
   # Reemplaza <POD_NAME> por el nombre obtenido en el paso anterior
   kubectl cp .output/checkpoint.pt tfm-slm/<POD_NAME>:/app/data/checkpoint.pt
   ```
3. Reinicia el pod para que lea el archivo recién copiado durante su inicio:
   ```bash
   kubectl rollout restart deployment/tfm-slm-chat -n tfm-slm
   ```

### 3. Exponer y Acceder al Chatbot (API + Web UI)
Una vez que el pod esté activo, expón el puerto de servicio localmente:

```bash
kubectl port-forward svc/tfm-slm-chat-service -n tfm-slm 8000:8000
```

Ahora puedes interactuar con el chatbot:

#### Opción 1: Interfaz Web Integrada
Abre tu navegador web e ingresa a:
[http://localhost:8000](http://localhost:8000)

Se abrirá una interfaz web moderna en modo oscuro que te permitirá hablar directamente con el chatbot en tiempo real.

#### Opción 2: Consultas directas vía API
Puedes hacer una consulta utilizando `curl` en tu terminal:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hola, ¿cómo funciona el Hybrid Block?"}'
```

---

## Cómo Parar y Volver a Desplegar la Aplicación

### Opción 1: Apagar y encender conservando tus datos (Recomendado)
Esta opción libera toda la memoria RAM y CPU de tu Mac en Kubernetes, pero **conserva el volumen persistente** para que no tengas que volver a copiar el archivo `checkpoint.pt` de 2 GB.

> [!IMPORTANT]
> **Nota sobre ArgoCD (GitOps):** Si despliegas mediante ArgoCD y tienes activada la opción de auto-curación (`selfHeal: true`), ArgoCD detectará que el pod se ha apagado y lo escalará de nuevo a 1 automáticamente. 
> Para poder pararlo usando ArgoCD debes usar una de estas vías:
> * **Método GitOps**: Modifica `spec.replicas: 0` en `manifests/chat-deployment.yaml` y haz commit en Git.
> * **Método Manual**: Desactiva temporalmente la opción **SelfHeal** en los ajustes de la app dentro de la interfaz web de ArgoCD antes de lanzar el comando `kubectl scale`.

*   **Para pararlo (Escalar a 0 pods):**
    ```bash
    kubectl scale deployment tfm-slm-chat -n tfm-slm --replicas=0
    ```
*   **Para volver a desplegarlo (Escalar a 1 pod):**
    ```bash
    kubectl scale deployment tfm-slm-chat -n tfm-slm --replicas=1
    ```

### Opción 2: Destruir y recrear todo desde cero
Esta opción elimina todos los servicios y recursos. **El volumen persistente se eliminará**, por lo que tendrás que volver a copiar el archivo `checkpoint.pt` de 2 GB cuando lo vuelvas a desplegar.

*   **Para destruirlo todo:**
    ```bash
    kubectl delete -k manifests/
    ```
*   **Para volver a crearlo todo:**
    ```bash
    kubectl apply -k manifests/
    ```

