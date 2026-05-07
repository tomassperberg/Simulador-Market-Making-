[README.md](https://github.com/user-attachments/files/27498932/README.md)
# Simulador-Market-Making-
Proyecto fin de curso Trading 
# Proyecto 03 — Análisis de Microestructura y Market Making
### Curso: Trading Cuantitativo · MFQ · 1er Bimestre 2026

---

## Descripción

Este proyecto implementa un **motor de simulación de market making** basado en el modelo óptimo de Avellaneda-Stoikov (2008), junto con un análisis completo de **microestructura de mercado** sobre datos sintéticos calibrados con procesos de Hawkes y movimiento browniano geométrico (GBM).

El proyecto forma parte del curso de Trading Cuantitativo y cubre desde la reconstrucción de un order book L2 hasta la evaluación de estrategias de provisión de liquidez bajo distintos parámetros de aversión al riesgo e horizonte temporal.

---

## Estructura del repositorio

```
.
├── simulador_market_making.py     # Motor de simulación A-S vs Naive MM
├── notebook_microestructura.ipynb # Análisis de microestructura (LOB L2, spreads, OFI)
├── README.md
└── requirements.txt
```

---

## Módulos principales (`simulador_market_making.py`)

| Módulo | Descripción |
|---|---|
| `OrderBook` | Libro de órdenes con heap bidireccional y lazy deletion |
| `AvellanedaStoikovMM` | Market maker óptimo según A-S 2008 (precio de reserva + spread óptimo) |
| `NaiveMM` | Benchmark con spread fijo simétrico |
| `SimulationEngine` | Loop event-driven tick-by-tick (resolución 1 segundo), proceso de Hawkes para llegada de órdenes |
| `SensitivityAnalyzer` | Grid de sensibilidad γ × T (6 × 3 = 18 combinaciones) |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd <nombre-del-repo>

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

**Requisito:** Python 3.9 o superior.

---

## Uso

### Simulación principal (A-S vs Naive MM)
```bash
python simulador_market_making.py
```

Simula una sesión completa de 6.5 horas (23 400 segundos), compara el market maker de Avellaneda-Stoikov contra el benchmark Naive, e imprime métricas de P&L, Sharpe y fills. Genera 4 gráficas interactivas.

### Tests unitarios (18 tests)
```bash
python simulador_market_making.py --test
```

Cubre `OrderBook`, `AvellanedaStoikovMM` y `SimulationEngine`. Todos deben pasar con `PASS`.

### Análisis de sensibilidad γ × T
```bash
python simulador_market_making.py --sensitivity
```

Recorre 18 combinaciones de aversión al riesgo (γ ∈ {0.01, 0.05, 0.10, 0.30, 0.50, 1.00}) y horizonte temporal (T ∈ {0.25, 0.50, 1.00}), imprime una tabla comparativa y genera heatmaps de P&L, Sharpe e inventario máximo.

### Notebook de microestructura
```bash
jupyter notebook notebook_microestructura.ipynb
```

Ejecutar las celdas en orden. La **Celda 1** instala dependencias automáticamente si es necesario.

---

## Parámetros clave

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `S0` | 100.0 | Precio inicial del activo ($) |
| `SIGMA_ANNUAL` | 0.25 | Volatilidad anual (25%) |
| `N_STEPS` | 23 400 | Duración de la sesión (segundos) |
| `GAMMA_DEFAULT` | 0.50 | Aversión al riesgo (γ) del A-S MM |
| `KAPPA_ARRIVAL` | 1.5 | Intensidad base de llegada de órdenes |
| `Q_MAX` | 8 | Inventario máximo permitido (lotes) |
| `LOT_SIZE` | 100 | Unidad mínima de orden (acciones) |
| `COMMISSION_PER_LOT` | $0.10 | Costo de transacción por lote ejecutado |

---

## Contenido del notebook (`notebook_microestructura.ipynb`)

1. Instalación y configuración de dependencias
2. Generación de datos sintéticos calibrados (GBM + Hawkes 6 procesos)
3. Reconstrucción y limpieza del Order Book L2 (5 niveles)
4. Análisis de spreads: Quoted, Effective y Realized
5. Profundidad del book por nivel
6. Patrones intradiarios de volatilidad y spread (U-shape)
7. Order Flow Imbalance (OFI)
8. Descomposición del spread — modelo Roll (1984)
9. Exportación de resultados

---

## Referencias

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.
- Cont, R., Kukanov, A. & Stoikov, S. (2014). *The price impact of order book events*. Journal of Financial Econometrics, 12(1), 47–88.
- Roll, R. (1984). *A simple implicit measure of the effective bid-ask spread in an efficient market*. Journal of Finance, 39(4), 1127–1139.
- Ogata, Y. (1981). *On Lewis' simulation method for point processes*. IEEE Transactions on Information Theory, 27(1), 23–31.

---

## Entregables del curso

| Entregable | Archivo | Descripción |
|---|---|---|
| E1 | `notebook_microestructura.ipynb` | Análisis de microestructura y LOB |
| E2 | `simulador_market_making.py` | Motor de simulación y estrategia A-S |
| E3 | Informe técnico (PDF) | 12–18 páginas con resultados y análisis |
| E4 | Video presentación | Defensa oral de 15 minutos |

**Fecha de entrega:** 30 de abril de 2026 · Grupos de máximo 4 integrantes.

---

## Reproducibilidad

El proyecto completo debe correr sin intervención manual. Para verificar:

```bash
# Tests (debe imprimir PASS)
python simulador_market_making.py --test

# Simulación completa
python simulador_market_making.py
```
