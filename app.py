import streamlit as st
import pandas as pd
from datetime import datetime
import os
import io
import base64
import html  # para escapar texto en la nota clínica

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

import sqlalchemy as sa
from sqlalchemy import text

# ============================
#   CONEXIÓN A BASE DE DATOS
# ============================

@st.cache_resource
def get_engine():
    """
    Crea y cachea el engine de SQLAlchemy usando la URL
    definida en st.secrets["db"]["url"].
    Además prueba la conexión y muestra el error real si falla.
    """
    try:
        db_url = st.secrets["db"]["url"]
    except Exception as e:
        st.error(
            "No se encontró st.secrets['db']['url']. "
            "Revisa los Secrets en Streamlit Cloud (Settings → Secrets).\n\n"
            f"Detalle: {e}"
        )
        st.stop()

    try:
        engine = sa.create_engine(db_url)

        # Probar conexión inmediatamente para obtener mensaje claro
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return engine
    except Exception as e:
        st.error(f"Error al conectar a la base de datos:\n\n{e}")
        st.stop()


# ============================
#       RUTAS DE ARCHIVOS
# ============================

# Dataset clínico encriptado (solo lectura)
ENC_CASES_PATH = "data/cliberto_qa_evalset_anon_992.enc"


# ============================
#    CIFRADO / DESCIFRADO
# ============================

def derive_key(password: str, salt: bytes) -> bytes:
    """Deriva una llave simétrica segura a partir de la contraseña y el salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
        backend=default_backend(),
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))


def decrypt_cases_from_file(password: str):
    """
    Desencripta el archivo .enc con la contraseña
    y regresa un DataFrame leído desde el XLSX original,
    adaptando columnas y calculando dificultad.
    """
    if not os.path.exists(ENC_CASES_PATH):
        raise FileNotFoundError(f"No se encontró el archivo encriptado: {ENC_CASES_PATH}")

    with open(ENC_CASES_PATH, "rb") as f:
        raw = f.read()

    salt = raw[:16]
    token = raw[16:]

    key = derive_key(password, salt)
    fernet = Fernet(key)

    try:
        decrypted = fernet.decrypt(token)
    except InvalidToken:
        return None

    buffer = io.BytesIO(decrypted)
    df = pd.read_excel(buffer)

    # Renombrar columnas del dataset nuevo a las esperadas por la app
    df = df.rename(columns={
        "id": "case_id",
        "context_anon": "nota_clinica",
        "question_anon": "pregunta",
        "pred_answer_anon": "respuesta_modelo",
        "pred_start": "answer_start",
        "pred_end": "answer_end",
    })

    # Clasificar dificultad según EM y F1
    def clasificar_dificultad(row):
        EM = row.get("EM", 0)
        F1 = row.get("F1", 0)

        try:
            EM = float(EM)
            F1 = float(F1)
        except Exception:
            EM = 0
            F1 = 0

        if EM == 1 and F1 == 1:
            return "Sencilla"
        elif EM == 0 and F1 > 0:
            return "Moderada"
        else:
            return "Difícil"

    df["dificultad"] = df.apply(clasificar_dificultad, axis=1)

    return df


# ============================
#       CARGA DE DATOS
# ============================

def load_cases():
    """Devuelve el DataFrame de casos desde la sesión."""
    if "cases_df" not in st.session_state:
        st.error("El dataset no ha sido cargado en la sesión.")
        st.stop()
    return st.session_state["cases_df"]


def load_users():
    """Carga usuarios desde la base de datos externa."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM usuarios", conn)
    except Exception as e:
        st.error(
            "Error al leer la tabla 'usuarios' desde la base de datos.\n\n"
            "Verifica que la tabla exista y que el usuario tenga permisos.\n\n"
            f"Detalle: {e}"
        )
        st.stop()

    if df.empty:
        return pd.DataFrame(columns=[
            "doctor_id", "password",
            "nombre", "apellido_paterno", "apellido_materno",
            "rol_profesional", "especialidad",
            "cedula", "edad", "institucion"
        ])

    return df


def save_users(df_new_user: pd.DataFrame):
    """
    Inserta SOLO los usuarios nuevos en la tabla usuarios.
    df_new_user debe ser un DataFrame con uno o varios registros nuevos.
    """
    engine = get_engine()
    try:
        with engine.begin() as conn:
            df_new_user.to_sql("usuarios", conn, if_exists="append", index=False)
    except Exception as e:
        st.error(
            "Error al guardar un nuevo usuario en la base de datos.\n\n"
            "Verifica que la tabla 'usuarios' exista y tenga las columnas correctas.\n\n"
            f"Detalle: {e}"
        )
        st.stop()


def load_annotations():
    """Carga anotaciones desde la base de datos externa."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM anotaciones", conn)
    except Exception as e:
        st.error(
            "Error al leer la tabla 'anotaciones' desde la base de datos.\n\n"
            "Verifica que la tabla exista y que el usuario tenga permisos.\n\n"
            f"Detalle: {e}"
        )
        st.stop()

    if df.empty:
        return pd.DataFrame(columns=[
            "doctor_id", "case_id", "pregunta",
            "respuesta_modelo", "correcta",
            "comentarios", "timestamp"
        ])

    return df


# ============================
#          UTILIDADES
# ============================

def highlight_span(text, start, end):
    """
    Resalta el span [start:end] en el texto con <mark>.
    Escapa HTML para no romper el frontend.
    Si no hay índices válidos, regresa el texto escapado tal cual.
    """
    try:
        if pd.isna(start) or pd.isna(end):
            return html.escape(text)

        start = int(start)
        end = int(end)
        if start < 0 or end > len(text) or start >= end:
            return html.escape(text)

        before = text[:start]
        middle = text[start:end]
        after = text[end:]

        before_esc = html.escape(before)
        middle_esc = html.escape(middle)
        after_esc = html.escape(after)

        return (
            before_esc
            + "<mark style='background-color: #fff3b0;'>"
            + middle_esc
            + "</mark>"
            + after_esc
        )
    except Exception:
        return html.escape(text)


def get_user_stats(doctor_id, annotations, total_cases):
    """Calcula estadísticas de anotaciones por usuario."""
    user_ann = annotations[annotations["doctor_id"] == doctor_id]
    casos_revisados = user_ann["case_id"].nunique()
    correctas = user_ann[user_ann["correcta"] == 1].shape[0]
    incorrectas = user_ann[user_ann["correcta"] == 0].shape[0]
    return casos_revisados, correctas, incorrectas


def calcular_siguiente_caso(doctor_id, cases, annotations):
    """Devuelve el índice del siguiente caso no revisado por el usuario."""
    total_cases = len(cases)
    user_ann = annotations[annotations["doctor_id"] == doctor_id]
    revisados = set(user_ann["case_id"].unique())
    next_idx = 0
    for i, row in cases.iterrows():
        if row["case_id"] not in revisados:
            next_idx = i
            break
        else:
            next_idx = min(i + 1, total_cases - 1)
    return next_idx


# ============================
#           APP
# ============================

def main():
    st.set_page_config(page_title="Validación QA Clínica", layout="wide")
    st.title("Interfaz de validación de respuestas clínicas")

    # ============================
    #   DESBLOQUEO DEL DATASET
    # ============================
    if "dataset_password_ok" not in st.session_state:
        st.session_state["dataset_password_ok"] = False

    if not st.session_state["dataset_password_ok"]:
        st.subheader("Acceso al conjunto de casos clínicos")

        with st.form("dataset_password_form"):
            pwd = st.text_input("Contraseña del dataset", type="password")
            submitted = st.form_submit_button("Desbloquear")

        if not submitted:
            st.info("Ingresa la contraseña del dataset para continuar.")
            st.stop()

        df_test = decrypt_cases_from_file(pwd)
        if df_test is None:
            st.error("Contraseña incorrecta para el dataset.")
            st.stop()

        st.session_state["dataset_password"] = pwd
        st.session_state["dataset_password_ok"] = True
        st.session_state["cases_df"] = df_test

    # Dataset ya cargado en sesión
    cases = load_cases()
    total_cases = len(cases)
    annotations = load_annotations()
    users = load_users()

    # ============================
    #      LOGIN / REGISTRO
    # ============================
    if "doctor_id" not in st.session_state:
        st.subheader("Acceso a la plataforma")

        modo = st.radio(
            "Selecciona una opción:",
            ["Soy usuario nuevo", "Ya estoy registrado"],
            horizontal=True
        )

        # -------- REGISTRO NUEVO USUARIO --------
        if modo == "Soy usuario nuevo":
            st.markdown("### Registro de nuevo usuario")

            with st.form("registro_form"):
                doctor_id = st.text_input("Usuario (ID para iniciar sesión)")
                password = st.text_input("Contraseña", type="password")

                col1, col2, col3 = st.columns(3)
                with col1:
                    nombre = st.text_input("Nombre")
                with col2:
                    apellido_paterno = st.text_input("Apellido paterno")
                with col3:
                    apellido_materno = st.text_input("Apellido materno")

                col4, col5 = st.columns(2)
                with col4:
                    edad = st.text_input("Edad")
                with col5:
                    rol_profesional = st.selectbox(
                        "Rol profesional",
                        [
                            "Médico general",
                            "Médico especialista",
                            "Enfermería",
                            "Otro personal de salud"
                        ]
                    )

                especialidad = st.text_input(
                    "Especialidad (si aplica)",
                    placeholder="Ejemplo: Medicina interna, Cardiología, etc."
                )
                cedula = st.text_input(
                    "Cédula profesional (opcional)",
                    placeholder="Opcional"
                )
                institucion = st.text_input(
                    "Institución / Hospital donde labora",
                    placeholder="Ejemplo: IMSS, ISSSTE, Secretaría de Salud, Hospital privado..."
                )

                submitted = st.form_submit_button("Registrar y entrar")

                if submitted:
                    if not doctor_id.strip() or not password.strip():
                        st.error("Usuario y contraseña son obligatorios.")
                        return
                    if not nombre.strip() or not apellido_paterno.strip():
                        st.error("Nombre y apellido paterno son obligatorios.")
                        return
                    if not edad.strip():
                        st.error("La edad es obligatoria.")
                        return
                    try:
                        edad_int = int(edad)
                        if edad_int <= 0:
                            raise ValueError
                    except ValueError:
                        st.error("La edad debe ser un número entero válido.")
                        return

                    doctor_id = doctor_id.strip()

                    if doctor_id in users["doctor_id"].astype(str).values:
                        st.error("Ese usuario ya existe. Usa otro ID o entra como 'Ya estoy registrado'.")
                        return

                    nuevo = pd.DataFrame([{
                        "doctor_id": doctor_id,
                        "password": password,
                        "nombre": nombre.strip(),
                        "apellido_paterno": apellido_paterno.strip(),
                        "apellido_materno": apellido_materno.strip(),
                        "rol_profesional": rol_profesional,
                        "especialidad": especialidad.strip(),
                        "cedula": cedula.strip(),
                        "edad": edad_int,
                        "institucion": institucion.strip()
                    }])

                    # Actualizar copia en memoria
                    users = pd.concat([users, nuevo], ignore_index=True)
                    # Guardar solo el nuevo usuario en la BD
                    save_users(nuevo)

                    nombre_completo = f"{nombre.strip()} {apellido_paterno.strip()} {apellido_materno.strip()}".strip()

                    st.session_state["doctor_id"] = doctor_id
                    st.session_state["doctor_nombre"] = nombre_completo
                    st.session_state["finished"] = False
                    st.session_state["case_idx"] = 0

                    st.success("Usuario registrado correctamente. Bienvenido.")
                    st.rerun()

            return

        # -------- LOGIN USUARIO EXISTENTE --------
        else:
            st.markdown("### Inicio de sesión")

            with st.form("login_form"):
                doctor_id = st.text_input("Usuario (ID)")
                password = st.text_input("Contraseña", type="password")
                submitted = st.form_submit_button("Entrar")

                if submitted:
                    if not doctor_id.strip() or not password.strip():
                        st.error("Ingresa usuario y contraseña.")
                        return

                    doctor_id = doctor_id.strip()
                    users = load_users()

                    if doctor_id not in users["doctor_id"].astype(str).values:
                        st.error("Este usuario no está registrado.")
                        return

                    user_row = users[users["doctor_id"].astype(str) == doctor_id].iloc[0]
                    if str(user_row["password"]) != password:
                        st.error("Contraseña incorrecta.")
                        return

                    nombre_completo = (
                        f"{user_row.get('nombre','')} "
                        f"{user_row.get('apellido_paterno','')} "
                        f"{user_row.get('apellido_materno','')}"
                    ).strip()

                    st.session_state["doctor_id"] = doctor_id
                    st.session_state["doctor_nombre"] = nombre_completo
                    st.session_state["finished"] = False

                    next_idx = calcular_siguiente_caso(doctor_id, cases, annotations)
                    st.session_state["case_idx"] = next_idx
                    st.rerun()

            return

    # ============================
    #      SESIÓN INICIADA
    # ============================
    doctor_id = st.session_state["doctor_id"]
    doctor_nombre = st.session_state.get("doctor_nombre", "")

    annotations = load_annotations()

    st.sidebar.write(f"👨‍⚕️ Usuario: **{doctor_id}**")
    if doctor_nombre:
        st.sidebar.write(f"Nombre: {doctor_nombre}")

    casos_revisados, correctas, incorrectas = get_user_stats(
        doctor_id, annotations, total_cases
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Estadísticas")
    st.sidebar.write(f"Casos revisados: **{casos_revisados} / {total_cases}**")
    st.sidebar.write(f"Marcadas correctas: **{correctas}**")
    st.sidebar.write(f"Marcadas incorrectas: **{incorrectas}**")

    terminar = st.sidebar.button("🚪 Terminar sesión", type="primary")

    if terminar:
        st.session_state["finished"] = True

    if st.session_state.get("finished", False):
        st.subheader("Resumen de tu sesión de validación")

        st.write(f"Has revisado **{casos_revisados}** de **{total_cases}** casos.")
        st.write(f"- Respuestas marcadas como **correctas**: {correctas}")
        st.write(f"- Respuestas marcadas como **incorrectas**: {incorrectas}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("🔄 Volver a la revisión"):
                st.session_state["finished"] = False
                next_idx = calcular_siguiente_caso(doctor_id, cases, annotations)
                st.session_state["case_idx"] = next_idx
                st.rerun()

        with col_b:
            if st.button("🔐 Cerrar sesión"):
                for key in ["doctor_id", "doctor_nombre", "case_idx", "finished"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        return

    # ============================
    #    INTERFAZ DE VALIDACIÓN
    # ============================

    if "case_idx" not in st.session_state:
        st.session_state["case_idx"] = 0

    idx = st.session_state["case_idx"]

    st.sidebar.markdown("---")
    st.sidebar.write(f"Caso actual: **{idx + 1} / {total_cases}**")

    nav_col1, nav_col2 = st.sidebar.columns(2)
    with nav_col1:
        if st.button("⬅️ Anterior", use_container_width=True):
            st.session_state["case_idx"] = max(0, idx - 1)
            st.rerun()
    with nav_col2:
        if st.button("Siguiente ➡️", use_container_width=True):
            st.session_state["case_idx"] = min(total_cases - 1, idx + 1)
            st.rerun()

    case = cases.iloc[idx]

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"Nota clínica (Caso {int(case['case_id'])})")
        nota = str(case["nota_clinica"])
        nota_html = highlight_span(
            nota,
            case.get("answer_start", None),
            case.get("answer_end", None)
        )
        st.markdown(
            f"<div style='white-space: pre-wrap; font-size: 14px;'>{nota_html}</div>",
            unsafe_allow_html=True
        )

    with col_right:
        st.subheader("Pregunta")
        st.info(str(case["pregunta"]))

        st.subheader("Respuesta del modelo")
        st.success(str(case["respuesta_modelo"]))

        dificultad = case.get("dificultad", "No disponible")
        st.markdown(f"**Dificultad estimada (EM/F1):** {dificultad}")

        st.markdown("---")
        if "clear_comment" not in st.session_state:
            st.session_state["clear_comment"] = False
        
        if st.session_state["clear_comment"]:
            st.session_state["comentarios_input"] = ""
            st.session_state["correcta_input"] = "Correcta"  # 👈 aquí está el truco
            st.session_state["clear_comment"] = False

        with st.form("annotation_form"):
            correcta = st.radio(
                "¿La respuesta del modelo es correcta?",
                options=["Correcta", "Incorrecta"],
                index=0, 
                horizontal=True,
                key="correcta_input"
            )
            comentarios = st.text_area(
                "Comentarios adicionales (opcional)",
                placeholder="Ejemplo: El diagnóstico correcto es...",
                key="comentarios_input"
            )

            guardar = st.form_submit_button("Guardar y pasar al siguiente caso")

            if guardar:
                nueva_fila = {
                    "doctor_id": doctor_id,
                    "case_id": case["case_id"],
                    "pregunta": case["pregunta"],
                    "respuesta_modelo": case["respuesta_modelo"],
                    "correcta": 1 if correcta == "Correcta" else 0,
                    "comentarios": comentarios,
                    "timestamp": datetime.now().isoformat()
                }

                annotations = pd.concat(
                    [annotations, pd.DataFrame([nueva_fila])],
                    ignore_index=True
                )

                engine = get_engine()
                try:
                    with engine.begin() as conn:
                        pd.DataFrame([nueva_fila]).to_sql(
                            "anotaciones",
                            conn,
                            if_exists="append",
                            index=False
                        )
                except Exception as e:
                    st.error(
                        "Error al guardar la anotación en la base de datos.\n\n"
                        "Verifica que la tabla 'anotaciones' exista y tenga las columnas correctas.\n\n"
                        f"Detalle: {e}"
                    )
                    st.stop()

                st.success("Respuesta registrada.")

                # 🔥 limpiar comentarios
                st.session_state["clear_comment"] = True

                if idx < total_cases - 1:
                    st.session_state["case_idx"] = idx + 1
                st.rerun()


if __name__ == "__main__":
    main()
