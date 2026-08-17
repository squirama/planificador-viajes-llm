# Planificador de viajes personalizado con LLM

Caso Práctico Unidad 1 — asignatura **Generative AI**, Máster en IA Aplicada (Instituto Europeo de
Posgrado). Enunciado completo en `../IEP-IAA-GIA_u1.pdf`.

## 1. Qué hace este proyecto

Una aplicación web (Streamlit) donde el usuario indica sus preferencias de viaje (destino, duración,
presupuesto, intereses, restricciones) y un LLM genera un itinerario día a día, apoyándose en una mini
base de conocimiento de destinos (RAG) y pasando por un control de calidad antes de mostrarse. El usuario
puede dar feedback y pedir que se regenere el itinerario con ese cambio.

El LLM de generación puede ser **Ollama** (modelo local, gratis, corre en tu GPU) o **Groq** (API en la
nube, gratis con API key), elegible con una variable de entorno — ver sección 6. Esto permite tanto
correr la app 100% local como desplegarla con una URL pública sin depender de tu PC.

## 2. Arquitectura

```
┌─────────────┐      ┌──────────────┐      ┌───────────────────┐
│  app.py     │──1──▶│  rag.py      │──2──▶│ data/guides/*.md   │
│ (Streamlit) │      │ (recupera    │      │ (base de           │
│             │      │  contexto)   │      │  conocimiento)     │
│             │◀──3──┤              │      └───────────────────┘
│             │      └──────────────┘
│             │──4──▶┌──────────────┐      ┌───────────────────┐
│             │      │  prompts.py  │──5──▶│ llm_client.py      │──▶ Ollama (local)
│             │◀──6──┤ (construye   │      │ (habla con Ollama  │    o Groq (nube)
│             │      │  el prompt)  │      │  o Groq según      │    según PROVEEDOR
│             │      └──────────────┘      │  config.PROVEEDOR) │
│             │                            └───────────────────┘
│             │──7──▶┌──────────────┐
│             │◀──8──┤  quality.py  │──▶ comprobaciones + auto-crítica del LLM
└─────────────┘      └──────────────┘
```

1. El usuario rellena el formulario → `app.py` arma un texto de consulta (destino + intereses).
2. `rag.py` busca en `data/guides/` los fragmentos más parecidos a esa consulta. Los embeddings se
   calculan en local con `sentence-transformers` (CPU, no depende de Ollama ni de Groq) y se comparan
   con similitud coseno.
3. Esos fragmentos vuelven como "contexto" en texto plano.
4. `app.py` pasa preferencias + contexto a `prompts.py`.
5. `prompts.py` arma el prompt final y `llm_client.py` se lo manda a Ollama o a Groq, según
   `config.PROVEEDOR`.
6. El itinerario generado vuelve a `app.py` y se muestra en pantalla.
7. Antes de darlo por bueno, `quality.py` revisa el itinerario.
8. Los avisos (si los hay) se muestran junto al itinerario.

## 3. Cómo se cubre cada requisito del enunciado

| # | Requisito | Cómo se cubre |
|---|-----------|----------------|
| 1 | Entrada de preferencias del usuario | Formulario en `app.py`: destino, duración, presupuesto, intereses, restricciones. |
| 2 | LLM previamente entrenado como base | Ollama (local) o Groq (nube) en `llm_client.py`, elegible con `config.PROVEEDOR`. |
| 3 | Técnicas de prompting | `prompts.py`: prompt de rol (system prompt), instrucciones de formato de salida, y refinamiento iterativo cuando hay feedback. |
| 4 | RAG con fuentes externas | `rag.py` + `data/guides/`: fichas de destino troceadas, convertidas a embeddings con `sentence-transformers` y recuperadas por similitud coseno según la consulta del usuario. |
| 5 | Fine-tuning en un dataset de itinerarios | **No implementado, documentado en la sección 4.** |
| 6 | GAN / RL (opcional) | **No implementado como entrenamiento real**, documentado en la sección 5. Sí se implementa un bucle de refinamiento con el feedback del usuario (ver `app.py`, botón "Regenerar con este feedback"), como aproximación conceptual ligera. |
| 7 | Filtrado y control de calidad | `quality.py`: comprobaciones automáticas (número de días, mención del destino) + auto-crítica pidiéndole al propio LLM que revise su respuesta. |
| 8 | Interfaz de usuario | `app.py` (Streamlit): formulario, itinerario, avisos de calidad, caja de feedback y regeneración. |

## 4. Diseño de fine-tuning (requisito 5, no implementado)

Entrenar de verdad un modelo fine-tuneado necesita un dataset curado y horas de GPU dedicadas, algo que
se sale del alcance razonable de un caso práctico de unidad. Aun así, así es como se abordaría:

- **Dataset**: pares `(preferencias del usuario) → (itinerario de alta calidad)`, recopilados de guías de
  viaje profesionales o generados y luego corregidos a mano. Formato tipo instrucción-respuesta (el mismo
  que usan la mayoría de frameworks de fine-tuning de LLMs).
- **Método**: LoRA o QLoRA sobre un modelo base open source (por ejemplo, Llama 3.1 8B, el mismo que
  usa este prototipo), que permite ajustar el modelo sin reentrenar todos sus parámetros y es viable en
  una sola GPU de consumo.
- **Herramientas**: frameworks como Unsloth o Axolotl para el entrenamiento, y luego exportar el modelo
  ajustado como un `Modelfile` de Ollama para poder usarlo igual que se usa `llama3.1:8b` en este
  proyecto (bastaría con cambiar `MODELO_GENERACION` en `config.py`).
- **Evaluación**: comparar itinerarios generados por el modelo base vs. el modelo ajustado sobre un
  conjunto de preferencias de prueba, con una mezcla de métricas automáticas (coherencia con las
  restricciones, mismo tipo de comprobaciones que hace `quality.py`) y valoración humana.
- **Por qué se sustituye por prompting + RAG en este prototipo**: con un buen system prompt y contexto
  recuperado por RAG, un modelo base ya genera itinerarios razonablemente buenos sin necesidad de
  entrenamiento adicional. El fine-tuning aportaría sobre todo consistencia de estilo y mejor manejo de
  casos límite, pero no es imprescindible para demostrar el concepto.

## 5. Diseño de GAN / RL (requisito 6, opcional, no implementado como entrenamiento real)

- **GANs**: están pensadas para generar datos continuos (imágenes, audio) donde un generador y un
  discriminador compiten. Para texto estructurado como un itinerario, no es la técnica habitual ni la
  más efectiva; los LLM actuales ya superan a los enfoques basados en GAN para este tipo de generación.
  Por eso no se ha implementado.
- **RL / RLHF**: sí tiene sentido conceptualmente — usar el feedback del usuario para mejorar las
  respuestas del modelo con el tiempo (ajustando una política mediante recompensas basadas en las
  valoraciones de los usuarios). Implementarlo de verdad requiere recolectar feedback de muchos usuarios,
  entrenar un modelo de recompensa y ajustar el LLM con un algoritmo como PPO — de nuevo, fuera de
  alcance para este prototipo.
- **Lo que sí se implementa como aproximación real**: cuando el usuario escribe feedback y pulsa
  "Regenerar con este feedback", `app.py` vuelve a construir el prompt incluyendo el itinerario anterior
  y el comentario del usuario (`construir_prompt_itinerario` en `prompts.py`). No es RLHF de verdad, pero
  es el mismo principio a nivel conceptual: el feedback humano cambia la siguiente generación.

## 6. Instalación y ejecución

### Opción A: local con Ollama (por defecto, gratis, necesita tu PC encendido)

```bash
# 1. Instalar dependencias de Python
pip install -r requirements.txt

# 2. Descargar el modelo de Ollama (una sola vez)
ollama pull llama3.1:latest

# 3. Arrancar la app
streamlit run app.py
```

Se abrirá en el navegador (normalmente `http://localhost:8501`). Hace falta tener Ollama instalado y
corriendo (`ollama serve`, aunque normalmente se inicia solo al instalar Ollama). La primera vez que
arranca, descarga el modelo de embeddings de `sentence-transformers` automáticamente (~80 MB).

### Opción B: con Groq (nube, gratis con API key, sirve para desplegar la app)

1. Crear una cuenta gratis en [console.groq.com](https://console.groq.com) y generar una API key.
2. Definir dos variables de entorno antes de arrancar (PowerShell):
   ```powershell
   $env:PROVEEDOR = "groq"
   $env:GROQ_API_KEY = "tu-api-key-aqui"
   streamlit run app.py
   ```
   Con esto la app ya no necesita Ollama corriendo — genera los itinerarios con Groq en la nube.

### Desplegar en Streamlit Community Cloud (URL pública, para que cualquiera la pruebe sin instalar nada)

1. Subir este proyecto a un repositorio de GitHub (ver sección 9).
2. Entrar en [share.streamlit.io](https://share.streamlit.io), conectar la cuenta de GitHub y seleccionar
   el repositorio, con `app.py` como archivo principal.
3. En **Settings → Secrets** del panel de la app, añadir:
   ```toml
   PROVEEDOR = "groq"
   GROQ_API_KEY = "tu-api-key-aqui"
   ```
4. Desplegar. Streamlit Cloud instala `requirements.txt` automáticamente y la app queda accesible en una
   URL pública tipo `https://tu-app.streamlit.app`.

## 7. Estructura de archivos

```
proyecto/
  app.py              # Interfaz Streamlit
  config.py           # Selección de proveedor (Ollama/Groq), nombres de modelo y constantes
  llm_client.py        # Llamadas al LLM: Ollama (local) o Groq (nube), según config.PROVEEDOR
  prompts.py           # Construcción de prompts
  rag.py                # Carga, embeddings (sentence-transformers) y búsqueda del RAG
  quality.py            # Comprobaciones de calidad + auto-crítica
  data/guides/*.md       # Base de conocimiento de ejemplo (6 destinos)
  requirements.txt
  README.md              # Esta memoria
```

## 8. Limitaciones conocidas

- La base de conocimiento del RAG son 6 fichas de ejemplo escritas a mano, no una fuente de datos real
  actualizada (en producción se conectaría a guías de viaje, reseñas o APIs externas).
- Los embeddings se recalculan en cada arranque de la app (aceptable con un corpus tan pequeño; con más
  documentos convendría guardarlos en disco o usar una vector DB).
- El control de calidad detecta problemas evidentes (número de días, destino mencionado, auto-crítica del
  propio LLM) pero no sustituye una revisión humana para un uso real.
- Fine-tuning y GAN/RL quedan documentados como diseño, no implementados (ver secciones 4 y 5).

## 9. Código fuente

Repositorio público: **https://github.com/squirama/planificador-viajes-llm**
