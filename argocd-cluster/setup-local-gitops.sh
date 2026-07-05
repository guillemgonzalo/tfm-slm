#!/usr/bin/env bash
#
# Automatiza la Opción B del README (GitOps simulado con ArgoCD en local, sin GitHub):
#   - Arranca git daemon en background
#   - Commitea cambios locales
#   - Instala ArgoCD en el clúster kind
#   - Despliega la Application de ArgoCD
#   - Copia el checkpoint local si existe
#   - Expone ArgoCD (8080) y el chatbot (8000) en background
#
# Uso:
#   ./argocd-cluster/setup-local-gitops.sh
#
# Para parar los port-forwards y el git daemon al terminar:
#   ./argocd-cluster/setup-local-gitops.sh stop

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_DAEMON_BASE="$(dirname "$REPO_ROOT")"
PID_DIR="$SCRIPT_DIR/.local-run"
mkdir -p "$PID_DIR"

GIT_DAEMON_PID_FILE="$PID_DIR/git-daemon.pid"
ARGOCD_PF_PID_FILE="$PID_DIR/argocd-portforward.pid"
CHAT_PF_PID_FILE="$PID_DIR/chat-portforward.pid"
LOG_DIR="$PID_DIR/logs"
mkdir -p "$LOG_DIR"

CLUSTER_NAME="tfm-cluster"
APP_NAMESPACE="tfm-slm"
ARGOCD_NAMESPACE="argocd"
CHECKPOINT_PATH="$REPO_ROOT/.output/checkpoint.pt"

log() { echo -e "\n\033[1;34m[setup]\033[0m $1"; }

stop_all() {
  log "Parando procesos en background..."
  for f in "$GIT_DAEMON_PID_FILE" "$ARGOCD_PF_PID_FILE" "$CHAT_PF_PID_FILE"; do
    if [[ -f "$f" ]]; then
      pid="$(cat "$f")"
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && log "Proceso $pid detenido ($f)"
      fi
      rm -f "$f"
    fi
  done
  log "Listo. (El clúster kind y ArgoCD siguen desplegados; usa 'kind delete cluster --name $CLUSTER_NAME' para destruir todo)."
  exit 0
}

if [[ "${1:-}" == "stop" ]]; then
  stop_all
fi

command -v kind >/dev/null || { echo "Falta 'kind'. Instálalo primero."; exit 1; }
command -v kubectl >/dev/null || { echo "Falta 'kubectl'. Instálalo primero."; exit 1; }
command -v git >/dev/null || { echo "Falta 'git'."; exit 1; }

# 1. Construir imagen Docker
log "Construyendo imagen Docker tfm-slm:latest..."
docker build -t tfm-slm:latest "$REPO_ROOT"

# 2. Clúster kind
if kind get clusters | grep -qx "$CLUSTER_NAME"; then
  log "Clúster kind '$CLUSTER_NAME' ya existe, reutilizando."
else
  log "Creando clúster kind '$CLUSTER_NAME'..."
  kind create cluster --name "$CLUSTER_NAME"
fi
log "Cargando imagen en el clúster..."
kind load docker-image tfm-slm:latest --name "$CLUSTER_NAME"

# 3. Namespace de la app
kubectl get namespace "$APP_NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$APP_NAMESPACE"

# 4. Git daemon en background
if [[ -f "$GIT_DAEMON_PID_FILE" ]] && kill -0 "$(cat "$GIT_DAEMON_PID_FILE")" 2>/dev/null; then
  log "git daemon ya corriendo (PID $(cat "$GIT_DAEMON_PID_FILE"))."
else
  log "Arrancando git daemon (base-path=$GIT_DAEMON_BASE)..."
  nohup git daemon --base-path="$GIT_DAEMON_BASE" --export-all --reuseaddr --verbose \
    > "$LOG_DIR/git-daemon.log" 2>&1 &
  echo $! > "$GIT_DAEMON_PID_FILE"
  sleep 1
fi

# 5. Commit local (si hay cambios)
cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  log "Haciendo commit local de cambios pendientes..."
  git add .
  git commit -m "Configure local GitOps sync"
else
  log "No hay cambios pendientes que commitear."
fi

# 6. Instalar ArgoCD
kubectl get namespace "$ARGOCD_NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$ARGOCD_NAMESPACE"
if ! kubectl get deployment argocd-server -n "$ARGOCD_NAMESPACE" >/dev/null 2>&1; then
  log "Instalando ArgoCD..."
  kubectl apply --server-side -n "$ARGOCD_NAMESPACE" -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
else
  log "ArgoCD ya instalado."
fi

log "Esperando a que argocd-server esté listo..."
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n "$ARGOCD_NAMESPACE"

# 7. Password admin
ADMIN_PASSWORD="$(kubectl -n "$ARGOCD_NAMESPACE" get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" 2>/dev/null | base64 -d || true)"

# 8. Desplegar Application
log "Aplicando Application de ArgoCD..."
kubectl apply -f "$SCRIPT_DIR/application.yaml"

# 9. Copiar checkpoint si existe
log "Esperando a que ArgoCD sincronice y cree el pod del chatbot..."
POD_APPEARED=false
for i in $(seq 1 60); do
  if kubectl get pod -l app=tfm-slm-chat -n "$APP_NAMESPACE" 2>/dev/null | grep -q tfm-slm-chat; then
    POD_APPEARED=true
    break
  fi
  sleep 5
done

if $POD_APPEARED && kubectl wait --for=condition=ready --timeout=180s pod -l app=tfm-slm-chat -n "$APP_NAMESPACE" 2>/dev/null; then
  POD_NAME="$(kubectl get pods -n "$APP_NAMESPACE" -l app=tfm-slm-chat -o jsonpath='{.items[0].metadata.name}')"
  if [[ -f "$CHECKPOINT_PATH" ]]; then
    log "Copiando checkpoint ($CHECKPOINT_PATH) al pod $POD_NAME..."
    kubectl cp "$CHECKPOINT_PATH" "$APP_NAMESPACE/$POD_NAME:/app/data/checkpoint.pt"
    kubectl rollout restart deployment/tfm-slm-chat -n "$APP_NAMESPACE"
    kubectl rollout status deployment/tfm-slm-chat -n "$APP_NAMESPACE" --timeout=180s
  else
    log "AVISO: no se encontró checkpoint en $CHECKPOINT_PATH. Cópialo manualmente con 'kubectl cp' cuando lo tengas."
  fi
else
  log "AVISO: el pod del chat no llegó a Ready a tiempo. Revisa 'kubectl get pods -n $APP_NAMESPACE'."
fi

# 10. Port-forwards en background
log "Exponiendo ArgoCD en https://localhost:8080..."
nohup kubectl port-forward svc/argocd-server -n "$ARGOCD_NAMESPACE" 8080:443 \
  > "$LOG_DIR/argocd-portforward.log" 2>&1 &
echo $! > "$ARGOCD_PF_PID_FILE"

log "Esperando a que exista el servicio tfm-slm-chat-service..."
for i in $(seq 1 30); do
  kubectl get svc tfm-slm-chat-service -n "$APP_NAMESPACE" >/dev/null 2>&1 && break
  sleep 2
done

log "Exponiendo el chatbot en http://localhost:8000..."
nohup kubectl port-forward svc/tfm-slm-chat-service -n "$APP_NAMESPACE" 8000:8000 \
  > "$LOG_DIR/chat-portforward.log" 2>&1 &
echo $! > "$CHAT_PF_PID_FILE"

sleep 2

cat <<EOF

=========================================================
 Todo listo.

 ArgoCD:   https://localhost:8080   (usuario: admin)
 Password: ${ADMIN_PASSWORD:-"<no disponible, ejecuta: kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d>"}

 Chatbot:  http://localhost:8000

 Para parar los port-forwards y el git daemon:
   $0 stop
=========================================================
EOF
