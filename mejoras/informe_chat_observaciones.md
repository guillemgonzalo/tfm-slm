# Informe de Observaciones — Chat tfm-slm

> Basado en las interacciones observadas durante la sesión de pruebas manual.
> El chat **no persiste historial** (solo DOM del navegador), por lo que este
> informe recoge las muestras reportadas en sesión, no un log exhaustivo.

## 1. Contexto del modelo evaluado

| Atributo | Valor |
|---|---|
| Arquitectura | Híbrida GRU + Flash Attention + MLP (12 bloques) |
| Parámetros | ~124M (hidden 768, 12 capas, 12 heads) |
| Tokenizer | GPT-2 (vocab 50257) |
| Contexto | 1024 tokens |
| Entrenamiento | Desde cero (no pre-entrenado) |
| Datos | Mix chat/instrucción **en inglés**, ~100k muestras |
| Generación | temperature=0.8, top_k=50, max_tokens=128 |

### Composición del dataset
| Dataset | Peso | Tipo |
|---|---|---|
| OpenAssistant (oasst1) | 30% | Diálogo asistente |
| UltraChat | 30% | Diálogo multivuelta |
| Alpaca | 20% | Instrucción → respuesta |
| ShareGPT | 20% | Conversaciones tipo ChatGPT |

Sin código (The Stack deshabilitado), sin dominio específico, sin español.

---

## 2. Interacciones observadas

### Interacción 1
- **Prompt:** `Give me tips to study`
- **Respuesta:** `Can you provide me with a list of any major variables that need to be considered for a study in Python?`
- **Análisis:**
  - ✅ Inglés gramaticalmente **coherente** (logro notable para modelo desde cero).
  - ❌ No responde: **repregunta** en lugar de dar contenido.
  - ❌ Deriva temática ("study" → "study in Python"), asociación superficial.

---

## 3. Diagnóstico

### 3.1 Lo que el modelo SÍ logra
- Coherencia **local** del inglés (sintaxis, registro conversacional).
- Imitación de **formato** de chat / instrucción.
- Frases plausibles y bien formadas a nivel de oración.

### 3.2 Lo que el modelo NO logra (y por qué)
| Síntoma | Causa raíz |
|---|---|
| Repregunta en vez de responder | No aprendió separación de rol `Assistant`; tras texto conversacional, generar otro turno `User:` es estadísticamente probable |
| Sin contenido factual/útil | Escala insuficiente: 124M params + ~100k muestras desde cero no codifican conocimiento |
| Deriva temática | Asociación léxica superficial sin razonamiento |
| Falla en español | Dataset 100% inglés |
| Sin memoria multivuelta | Prompt single-turn; `generate` no inyecta historial |

### 3.3 Causa estructural
La combinación **desde-cero + 124M + ~100k muestras + 1 época** produce un modelo
que aprende **forma**, no **contenido**. Referencia: GPT-2 (mismo tamaño) vio ~40 GB
de texto; aquí el orden de magnitud de datos es mucho menor.

---

## 4. Mejoras propuestas

### 4.1 Sin reentrenar (bajo coste, mejora la demo)
1. **Formato de prompt Alpaca** (alinea con 20% del training):
   ```
   ### Instruction:
   {pregunta}

   ### Response:
   ```
2. **Corte de generación** en `\nUser:` / `###` para frenar derivas a nuevo turno.
3. **Sampling más conservador:** temperature 0.6, top_k 30.
4. **Repetition penalty** para evitar bucles.

### 4.2 Con reentrenamiento (mejora real de calidad)
- Más muestras y más épocas.
- Dataset en español si el objetivo es responder en español.
- Filtrado de calidad / deduplicación.
- Considerar fine-tuning sobre base pre-entrenada en vez de desde cero.

### 4.3 Optimización ya aplicada
- **KV-cache** en `generate`: inferencia O(n²) → O(n), ~10-30x más rápida por
  respuesta en servicio persistente. Verificada equivalencia numérica (diff 1.6e-7).
- Fix de bug en fallback SDPA (máscara causal invertida en CPU/Mac).

---

## 5. Encuadre honesto para el TFM
El valor defendible del proyecto está en:
- **Arquitectura híbrida** GRU + Attention + MLP y su análisis de contribución por componente.
- **Pipeline completo** (download → process → train → serve) y despliegue GitOps (ArgoCD/Kind).
- **Optimizaciones de inferencia** (KV-cache, Flash Attention, bf16, torch.compile).

La precisión factual **no** es un objetivo realista a esta escala y debe presentarse
como limitación conocida, no como fallo.

---

## 6. Nota sobre captura de datos
Para informes futuros con historial real, añadir persistencia en `app/chat/api.py`
(guardar cada par prompt/respuesta a JSONL). Actualmente las conversaciones se
pierden al cerrar el navegador.
