import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Credit Risk Professional Suite", 
    page_icon="🏦",
    layout="wide"
)

# 1. CARGA DEL MODELO (Cache para optimizar carga)
@st.cache_resource
def load_model():
    # Asegúrate de que este archivo esté en la misma carpeta
    return joblib.load('model_lgbm_monotonic.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}. Asegúrate de que 'model_lgbm_monotonic.pkl' esté en la misma carpeta.")
    st.stop()

# --- INTERFAZ ---
st.title("🏦 Simulador de Riesgo Crediticio Profesional")
st.markdown("""
Esta herramienta evalúa la probabilidad de default utilizando un modelo de **Machine Learning (LightGBM)** con restricciones de monotonía, permitiendo análisis de resiliencia bajo escenarios de crisis.
""")

# 2. SIDEBAR PARA ENTRADAS (INPUTS)
st.sidebar.header("📊 Datos del Cliente")

limit_bal = st.sidebar.slider("Límite de Crédito (USD)", 1000, 1000000, 50000, step=1000)
age = st.sidebar.number_input("Edad", 18, 100, 30)
sex = st.sidebar.selectbox("Género", [1, 2], format_func=lambda x: "Hombre" if x==1 else "Mujer")
education = st.sidebar.selectbox("Educación", [1, 2, 3, 4], format_func=lambda x: ["Posgrado", "Universidad", "Bachillerato", "Otros"][x-1])
marriage = st.sidebar.selectbox("Estado Civil", [1, 2, 3], format_func=lambda x: ["Casado", "Soltero", "Otros"][x-1])

st.sidebar.subheader("🕒 Comportamiento de Pago")
pay_0 = st.sidebar.slider("Estado último pago (Mes actual)", -1, 8, 0)
max_delay = st.sidebar.slider("Máximo meses de retraso detectado", 0, 8, 0)

st.sidebar.subheader("💳 Utilización y Gastos")
total_bill = st.sidebar.number_input("Saldo Facturado Mes Actual (USD)", 0, 1000000, 5000)
avg_bill_6m = st.sidebar.number_input("Promedio Facturado últimos 6 meses (USD)", 0, 1000000, 5000)
utilization = st.sidebar.slider("Ratio de Utilización (0 a 1.5)", 0.0, 1.5, 0.3)
pay_ratio = st.sidebar.slider("Ratio de Pago (Pagado / Facturado)", 0.0, 1.0, 0.5)

# --- CÁLCULO DE FEATURES ELITE (BASE) ---
spending_velocity = total_bill / (avg_bill_6m + 1)

# --- 3. PROCESAMIENTO ESCENARIO BASE ---
# El orden de las columnas debe ser IDÉNTICO al entrenamiento
FEATURES_ORDER = [
    'LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
    'TOTAL_BILL_AMOUNT', 'MAX_DELAY', 'UTILIZATION_RATIO', 
    'PAYMENT_TO_BILL_RATIO', 'AVG_BILL', 'SPENDING_VELOCITY'
]

data_base = {
    'LIMIT_BAL': float(limit_bal),
    'AGE': float(age),
    'SEX': float(sex),
    'EDUCATION': float(education),
    'MARRIAGE': float(marriage),
    'PAY_0': float(pay_0),
    'PAY_2': 0.0, 'PAY_3': 0.0, 'PAY_4': 0.0, 'PAY_5': 0.0, 'PAY_6': 0.0, # Proxies
    'TOTAL_BILL_AMOUNT': float(total_bill),
    'MAX_DELAY': float(max_delay),
    'UTILIZATION_RATIO': float(utilization),
    'PAYMENT_TO_BILL_RATIO': float(pay_ratio),
    'AVG_BILL': float(avg_bill_6m),
    'SPENDING_VELOCITY': float(spending_velocity)
}

input_df = pd.DataFrame([data_base])[FEATURES_ORDER]

# Predicción base
prob_base = model.predict_proba(input_df)[:, 1][0]
el_base = prob_base * total_bill * 0.75 # LGD del 75%

# --- 4. SECCIÓN DE STRESS TESTING ---
st.sidebar.markdown("---")
st.sidebar.header("🌪️ Escenario de Stress Test")
stress_mode = st.sidebar.checkbox("Activar Escenario de Crisis")

if stress_mode:
    st.sidebar.warning("Escenario de Recesión Activado")
    
    # Aplicar stress: +30% en saldos y +1 mes en atraso
    input_df_stress = input_df.copy()
    input_df_stress['TOTAL_BILL_AMOUNT'] *= 1.3
    input_df_stress['PAY_0'] = input_df_stress['PAY_0'].apply(lambda x: min(x + 1, 8))
    input_df_stress['MAX_DELAY'] = input_df_stress['MAX_DELAY'].apply(lambda x: min(x + 1, 8))
    
    # Recalcular Velocity bajo Stress
    input_df_stress['SPENDING_VELOCITY'] = input_df_stress['TOTAL_BILL_AMOUNT'] / (avg_bill_6m + 1)
    
    # Predicción estresada
    prob_final = model.predict_proba(input_df_stress)[:, 1][0]
    total_bill_stress = total_bill * 1.3
    el_final = prob_final * total_bill_stress * 0.75
else:
    prob_final = prob_base
    el_final = el_base

# --- 5. RENDERIZADO DE RESULTADOS ---
st.subheader("Análisis de Riesgo y Pérdida Esperada")
col1, col2, col3 = st.columns(3)

with col1:
    delta_prob = (prob_final - prob_base) if stress_mode else None
    st.metric(
        label="Probabilidad de Default", 
        value=f"{prob_final:.2%}",
        delta=f"{delta_prob:.2%}" if delta_prob else None,
        delta_color="inverse"
    )

with col2:
    if prob_final > 0.30:
        status, color = "ALTO", "red"
    elif prob_final > 0.10:
        status, color = "MEDIO", "orange"
    else:
        status, color = "BAJO", "green"
    
    st.markdown(f"**Rating de Riesgo:**")
    st.markdown(f"<h2 style='color: {color};'>{status}</h2>", unsafe_allow_html=True)

with col3:
    delta_el = (el_final - el_base) if stress_mode else None
    st.metric(
        label="Pérdida Esperada (EL)", 
        value=f"${el_final:,.2f}",
        delta=f"${delta_el:,.2f}" if delta_el else None,
        delta_color="inverse"
    )

# Visualización de la métrica de Velocidad
st.write(f"### Dinámica de Consumo")
velocity_val = input_df_stress['SPENDING_VELOCITY'][0] if stress_mode else spending_velocity
st.write(f"Velocidad de Gasto actual: **{velocity_val:.2f}x** respecto al promedio histórico.")
st.progress(min(float(prob_final), 1.0))

# --- DETALLE TÉCNICO ---
with st.expander("🛠️ Ver Auditoría Técnica de Datos"):
    st.write("Datos enviados al modelo (Features finales):")
    st.dataframe(input_df_stress if stress_mode else input_df)
    st.info("Nota: PAY_2 a PAY_6 se completan con 0 para mantener la compatibilidad del tensor de entrada.")


st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center;'>
        <p style='margin-bottom: 0px;'>Desarrollado por:</p>
        <h3 style='margin-top: 0px;'>Alvarez Lucas</h3>
        <a href='https://www.linkedin.com/in/lucas-alvarez-2b7b092b1/' target='_blank'>
            <img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' width='100%'>
        </a>
    </div>
    """, 
    unsafe_allow_html=True
)


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        <p><strong>Metodología:</strong> Modelo LightGBM con restricciones de monotonía entrenado con el dataset UCI Credit Card Default. 
        Las métricas de Stress Test son simulaciones de escenarios macroeconómicos adversos.</p>
        <p>© 2024 - Credit risk analytics</p>
    </div>
    """, 
    unsafe_allow_html=True
)