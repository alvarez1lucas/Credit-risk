# 🏦 Credit Risk Engine & Stress Test Simulator

Este proyecto presenta una solución integral de **Machine Learning** para la evaluación del riesgo crediticio, diseñada para predecir la probabilidad de impago (*Default*) y analizar la resiliencia financiera bajo escenarios macroeconómicos adversos.

---

##  Live Demo
Puedes interactuar con el simulador en tiempo real aquí:
https://credit-risk-analytics-alvarez-lucas.streamlit.app/

---

##  Características Técnicas

### 1. Modelo de Machine Learning
* **Algoritmo:** LightGBM (Gradient Boosting Machine).
* **Optimización:** Implementación de **Monotonic Constraints** para asegurar que variables críticas (como el historial de retrasos) tengan una relación lógica y consistente con el riesgo, aumentando la interpretabilidad y confiabilidad del modelo en producción.
* **Métrica Objetivo:** Probabilidad de Default (PD).

### 2. Feature Engineering Avanzado
Se desarrollaron variables financieras clave para capturar el comportamiento dinámico del cliente:
* **Spending Velocity:** Mide la aceleración del gasto del último mes vs. el promedio histórico.
* **Utilization Ratio:** Nivel de uso del límite de crédito disponible.
* **Payment-to-Bill Ratio:** Capacidad de pago real frente a la facturación.

### 3. Módulo de Stress Testing & Pérdida Esperada (EL)
El simulador permite estresar la cartera aumentando los saldos y los meses de mora simultáneamente, recalculando en tiempo real:
* **Δ PD:** El incremento en la probabilidad de impago.
* **Expected Loss (EL):** Basado en la fórmula $EL = PD \times EAD \times LGD$.

---

##  Tecnologías Utilizadas
* **Lenguaje:** Python 3.11
* **Librerías:** Pandas, Numpy, Scikit-learn, LightGBM, Joblib.
* **Despliegue:** Streamlit Cloud.


*Nota: Este proyecto utiliza el dataset UCI Credit Card Default para fines analíticos y educativos.*
