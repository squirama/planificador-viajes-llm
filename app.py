# Interfaz del planificador de viajes (requisitos 1 y 8 del enunciado).
# Para arrancarla: streamlit run app.py

import streamlit as st
from prompts import SYSTEM_PROMPT, construir_prompt_itinerario
from llm_client import generar_texto
from rag import cargar_fragmentos, calcular_embeddings, buscar_contexto
from quality import revisar_itinerario

st.set_page_config(page_title="Planificador de viajes con IA", page_icon="🧳")
st.title("🧳 Planificador de viajes personalizado")
st.caption("Caso Práctico Unidad 1 - Generative AI (IEP)")


@st.cache_resource
def preparar_base_conocimiento():
    """Carga las guías y calcula sus embeddings una sola vez (se cachea entre interacciones)."""
    fragmentos = cargar_fragmentos()
    embeddings = calcular_embeddings(fragmentos)
    return fragmentos, embeddings


fragmentos, embeddings_fragmentos = preparar_base_conocimiento()

# --- Formulario de preferencias (requisito 1) ---
with st.form("formulario_viaje"):
    destino = st.text_input("Destino", placeholder="Ej: Kyoto, Japón")
    col1, col2 = st.columns(2)
    duracion = col1.number_input("Duración (días)", min_value=1, max_value=30, value=5)
    presupuesto = col2.number_input("Presupuesto total (€)", min_value=0, value=1000)
    intereses = st.multiselect(
        "Intereses",
        ["Naturaleza", "Cultura e historia", "Gastronomía", "Aventura", "Vida nocturna", "Compras", "Relax"],
    )
    restricciones = st.text_area("Restricciones o requisitos especiales (opcional)")
    enviado = st.form_submit_button("Generar itinerario")

if "itinerario" not in st.session_state:
    st.session_state.itinerario = None
    st.session_state.preferencias = None

if enviado:
    if not destino or not intereses:
        st.error("Por favor, indica al menos el destino y un interés.")
    else:
        preferencias = {
            "destino": destino,
            "duracion": int(duracion),
            "presupuesto": presupuesto,
            "intereses": intereses,
            "restricciones": restricciones,
        }
        with st.spinner("Buscando información del destino (RAG)..."):
            consulta = f"{destino} {' '.join(intereses)}"
            contexto = buscar_contexto(consulta, fragmentos, embeddings_fragmentos)

        with st.spinner("Generando itinerario con el LLM..."):
            prompt = construir_prompt_itinerario(preferencias, contexto)
            itinerario = generar_texto(prompt, system=SYSTEM_PROMPT)

        st.session_state.itinerario = itinerario
        st.session_state.preferencias = preferencias

# --- Mostrar itinerario generado (requisito 8) ---
if st.session_state.itinerario:
    st.subheader("Itinerario propuesto")
    st.markdown(st.session_state.itinerario)

    with st.spinner("Revisando calidad del itinerario..."):
        avisos = revisar_itinerario(st.session_state.preferencias, st.session_state.itinerario)

    if avisos:
        st.warning("Avisos de calidad detectados:\n\n" + "\n".join(f"- {a}" for a in avisos))
    else:
        st.success("El itinerario ha pasado los controles de calidad.")

    # --- Feedback y regeneración (requisitos 6 y 8) ---
    st.subheader("¿Quieres cambiar algo?")
    feedback = st.text_area("Escribe qué te gustaría modificar", key="feedback")
    if st.button("Regenerar con este feedback") and feedback:
        with st.spinner("Regenerando itinerario con tu feedback..."):
            prompt = construir_prompt_itinerario(
                st.session_state.preferencias,
                contexto_rag="",
                itinerario_anterior=st.session_state.itinerario,
                feedback=feedback,
            )
            nuevo_itinerario = generar_texto(prompt, system=SYSTEM_PROMPT)
        st.session_state.itinerario = nuevo_itinerario
        st.rerun()
