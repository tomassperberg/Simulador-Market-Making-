"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Proyecto 03 — Motor de Simulación Market Making (Avellaneda-Stoikov 2008)  ║
║  Curso: Trading Cuantitativo · MFQ 1er Bimestre 2026                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Uso:
    python simulador_market_making.py              → simulación principal
    python simulador_market_making.py --test       → 18 tests unitarios
    python simulador_market_making.py --sensitivity → análisis γ × T

Módulos:
    OrderBook            — Libro de órdenes con heap bidireccional
    AvellanedaStoikovMM  — Market maker óptimo (A-S 2008)
    NaiveMM              — Benchmark: spread fijo simétrico
    SimulationEngine     — Loop event-driven, registro de P&L
    SensitivityAnalyzer  — Grid γ × T

Dependencias: numpy, pandas, pyarrow, matplotlib
"""

from __future__ import annotations
import argparse
import heapq
import math
import time
import unittest
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# 0. PARÁMETROS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

RNG_SEED    = 42

# ── Activo ────────────────────────────────────────────────────────────────────
S0          = 100.0            # Precio inicial ($)
SIGMA_ANNUAL= 0.25             # Volatilidad anual
TICK        = 0.01             # Tamaño mínimo de tick
LOT_SIZE    = 100              # Unidad mínima de orden (acciones)

# ── Sesión ────────────────────────────────────────────────────────────────────
N_STEPS     = 23_400           # 9:30–16:00 = 23 400 segundos
DT          = 1.0              # Resolución temporal (1s)
T_SECONDS   = float(N_STEPS * DT) # Segundos totales para fórmulas A-S

# ── Parámetros A-S ────────────────────────────────────────────────────────────
GAMMA_DEFAULT = 0.50           # Aversión al riesgo (γ)
KAPPA_ARRIVAL = 1.5            # Intensidad base de llegada de órdenes (λ₀, ord/s)
KAPPA_MO      = 1.5            # Decaimiento del fill probability (κ, ord/s)
T_DEFAULT     = 1.0            # Horizonte normalizado (T)
Q_MAX         = 8              # Inventario máximo (lotes)
COMMISSION_PER_LOT = 0.10      # Costo por lote ejecutado ($): comisión broker + market impact estimado

# ── Parámetros Hawkes ─────────────────────────────────────────────────────────
HAWKES_ALPHA  = 0.6            # Salto en intensidad tras un fill
HAWKES_BETA   = 1.5            # Tasa de decaimiento de la excitación

# Volatilidad relativa y absoluta por segundo
SIGMA_S = SIGMA_ANNUAL / math.sqrt(252 * N_STEPS)  # Relativa (para GBM)
SIGMA_ABS = S0 * SIGMA_S                           # Absoluta $/acción/√segundo (para A-S)
GAMMA_SCALE = 1.0 / LOT_SIZE                       # Escala de aversión al riesgo por lote

# ── Gráficas ──────────────────────────────────────────────────────────────────
COLORS = {'teal':'#34d399','blue':'#4f8cff','gold':'#f5c842',
          'red':'#f87171','purple':'#a78bfa','muted':'#7a8099'}

plt.rcParams.update({
    'figure.facecolor':'#0b0e14','axes.facecolor':'#111520',
    'axes.edgecolor':'#3a3f52','axes.labelcolor':'#e8eaf0',
    'text.color':'#e8eaf0','xtick.color':'#7a8099','ytick.color':'#7a8099',
    'grid.color':'#1e2333','grid.linestyle':'--','grid.alpha':0.5,
    'legend.facecolor':'#161b28','legend.edgecolor':'#3a3f52',
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Order:
    order_id : int
    side     : str
    price    : float
    qty      : int
    ts       : float
    active   : bool = True

@dataclass
class Trade:
    ts          : float
    side        : str
    price       : float
    qty         : int
    inventory   : int
    pnl_cum     : float

@dataclass
class Snapshot:
    ts           : float
    mid_price    : float
    bid_quote    : float
    ask_quote    : float
    spread_quoted: float
    reservation  : float
    inventory    : int
    pnl_mark     : float
    pnl_spread   : float
    pnl_costs    : float
    cash         : float

# ══════════════════════════════════════════════════════════════════════════════
# 2. ORDER BOOK
# ══════════════════════════════════════════════════════════════════════════════

class OrderBook:
    """Libro de órdenes con heap bidireccional y lazy deletion."""

    def __init__(self):
        self._bid_heap: List[Tuple[float, int, Order]] = []
        self._ask_heap: List[Tuple[float, int, Order]] = []
        self._orders: dict[int, Order] = {}
        self._next_id = 0
        self._trades: List[Trade] = []

    def place_limit(self, side: str, price: float, qty: int, ts: float) -> int:
        oid = self._next_id; self._next_id += 1
        order = Order(oid, side, price, qty, ts)
        self._orders[oid] = order
        if side == 'bid':
            heapq.heappush(self._bid_heap, (-price, oid, order))
        else:
            heapq.heappush(self._ask_heap, (price, oid, order))
        return oid

    def cancel(self, order_id: int) -> bool:
        order = self._orders.get(order_id)
        if order and order.active:
            order.active = False
            return True
        return False

    def _clean_heap(self, heap: list) -> None:
        while heap and not heap[0][2].active:
            heapq.heappop(heap)

    def best_bid(self) -> Optional[float]:
        self._clean_heap(self._bid_heap)
        return -self._bid_heap[0][0] if self._bid_heap else None

    def best_ask(self) -> Optional[float]:
        self._clean_heap(self._ask_heap)
        return self._ask_heap[0][0] if self._ask_heap else None

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2 if bb and ba else None

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return ba - bb if bb and ba else None

    def depth(self, side: str, n_levels: int = 1) -> float:
        heap  = self._bid_heap if side == 'bid' else self._ask_heap
        sign  = -1 if side == 'bid' else 1
        level_prices: dict[float, int] = {}
        for key, oid, order in heap:
            if not order.active:
                continue
            p = sign * key
            level_prices[p] = level_prices.get(p, 0) + order.qty
            if len(level_prices) >= n_levels:
                break
        return sum(level_prices.values())

    def snapshot(self, ts: float) -> dict:
        return {
            'ts'      : ts,
            'bid1'    : self.best_bid(),
            'ask1'    : self.best_ask(),
            'mid'     : self.mid_price(),
            'spread'  : self.spread(),
            'bid_depth': self.depth('bid'),
            'ask_depth': self.depth('ask'),
        }

# ══════════════════════════════════════════════════════════════════════════════
# 3. MARKET MAKER AVELLANEDA-STOIKOV (2008)
# ══════════════════════════════════════════════════════════════════════════════

class AvellanedaStoikovMM:
    def __init__(self, gamma: float = GAMMA_DEFAULT,
                 kappa: float = KAPPA_MO,
                 sigma: float = SIGMA_ABS, 
                 T: float = T_DEFAULT,
                 q_max: int = Q_MAX,
                 tick: float = TICK,
                 lot_size: int = LOT_SIZE):
        self.gamma        = gamma
        self.gamma_eff    = gamma * GAMMA_SCALE
        self.kappa        = kappa
        self.sigma        = sigma
        self.T            = T
        self.q_max        = q_max
        self.tick         = tick
        self.lot_size     = lot_size

        self.inventory   = 0
        self.cash        = 0.0
        self.pnl_cum     = 0.0
        self.pnl_spread  = 0.0
        self.total_costs = 0.0
        self.bid_order_id: Optional[int] = None
        self.ask_order_id: Optional[int] = None
        self._snapshots: List[Snapshot] = []
        self._n_fills_bid = 0
        self._n_fills_ask = 0

    def reservation_price(self, s: float, q: int, t: float) -> float:
        tau_norm = max(self.T - t, 1e-9)
        tau_s = tau_norm * T_SECONDS 
        q_acc = q * self.lot_size
        return s - q_acc * self.gamma_eff * (self.sigma ** 2) * tau_s

    def optimal_spread(self, t: float) -> float:
        tau_norm  = max(self.T - t, 1e-9)
        tau_s = tau_norm * T_SECONDS 
        term1 = self.gamma_eff * (self.sigma ** 2) * tau_s
        term2 = (2 / self.gamma_eff) * math.log(1 + self.gamma_eff / self.kappa)
        delta = term1 + term2
        return max(delta, self.tick)

    def quotes(self, s: float, t: float) -> Tuple[float, float]:
        q  = self.inventory
        r  = self.reservation_price(s, q, t)
        d  = self.optimal_spread(t)
        half = d / 2

        skew_factor = (q / self.q_max) * 0.5 * half if self.q_max > 0 else 0.0
        bid = r - half - skew_factor
        ask = r + half - skew_factor

        bid = round(bid / self.tick) * self.tick
        ask = round(ask / self.tick) * self.tick

        if ask - bid < self.tick:
            ask = bid + self.tick

        return bid, ask

    def mark_to_market(self, s: float) -> float:
        return self.cash + self.inventory * self.lot_size * s

    def on_fill(self, side: str, price: float, qty: int, mid_price: float) -> None:
        lots = qty // self.lot_size
        
        if side == 'bid':
            lots = min(lots, self.q_max - self.inventory)
        else:
            lots = min(lots, self.q_max + self.inventory)
            
        if lots <= 0:
            return
        
        cost = lots * COMMISSION_PER_LOT
        actual_qty = lots * self.lot_size  
        self.total_costs += cost
        
        # ── CÁLCULO REAL DEL CASH Y EDGE ──────────────────────────────
        if side == 'bid':
            self.inventory += lots
            self.cash      -= price * actual_qty  
            edge            = (mid_price - price) * actual_qty 
            self._n_fills_bid += 1
        else:
            self.inventory -= lots
            self.cash      += price * actual_qty  
            edge            = (price - mid_price) * actual_qty 
            self._n_fills_ask += 1
            
        self.pnl_spread += edge
        self.inventory = int(np.clip(self.inventory, -self.q_max, self.q_max))

    def record_snapshot(self, ts: float, s: float, t: float) -> None:
        bid, ask = self.quotes(s, t)
        r = self.reservation_price(s, self.inventory, t)
        snap = Snapshot(
            ts=ts, mid_price=s, bid_quote=bid, ask_quote=ask,
            spread_quoted=ask - bid, reservation=r,
            inventory=self.inventory,
            pnl_mark=self.mark_to_market(s),
            pnl_spread=self.pnl_spread,
            pnl_costs=self.total_costs,
            cash=self.cash
        )
        self._snapshots.append(snap)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(s) for s in self._snapshots])

# ══════════════════════════════════════════════════════════════════════════════
# 4. MARKET MAKER NAIVE (BENCHMARK)
# ══════════════════════════════════════════════════════════════════════════════

class NaiveMM:
    FIXED_SPREAD = 2 * TICK

    def __init__(self, q_max: int = Q_MAX, tick: float = TICK,
                 lot_size: int = LOT_SIZE):
        self.q_max     = q_max
        self.tick      = tick
        self.lot_size  = lot_size
        self.inventory = 0
        self.cash      = 0.0
        self.pnl_spread  = 0.0
        self.total_costs = 0.0
        self._snapshots: List[Snapshot] = []
        self._n_fills_bid = 0
        self._n_fills_ask = 0

    def quotes(self, s: float, t: float) -> Tuple[float, float]:
        half = self.FIXED_SPREAD / 2
        bid  = round((s - half) / self.tick) * self.tick
        ask  = round((s + half) / self.tick) * self.tick
        return bid, ask

    def reservation_price(self, s: float, *args, **kwargs) -> float:
        return s

    def mark_to_market(self, s: float) -> float:
        return self.cash + self.inventory * self.lot_size * s

    def on_fill(self, side: str, price: float, qty: int, mid_price: float) -> None:
        lots = qty // self.lot_size
        
        if side == 'bid':
            lots = min(lots, self.q_max - self.inventory)
        else:
            lots = min(lots, self.q_max + self.inventory)
            
        if lots <= 0:
            return
            
        cost = lots * COMMISSION_PER_LOT
        actual_qty = lots * self.lot_size 
        self.total_costs += cost
        
        # ── CÁLCULO REAL DEL CASH Y EDGE ──────────────────────────────
        if side == 'bid':
            self.inventory += lots
            self.cash      -= price * actual_qty 
            edge            = (mid_price - price) * actual_qty
            self._n_fills_bid += 1
        else:
            self.inventory -= lots
            self.cash      += price * actual_qty 
            edge            = (price - mid_price) * actual_qty
            self._n_fills_ask += 1
            
        self.pnl_spread += edge
        self.inventory = int(np.clip(self.inventory, -self.q_max, self.q_max))

    def record_snapshot(self, ts: float, s: float, t: float) -> None:
        bid, ask = self.quotes(s, t)
        snap = Snapshot(
            ts=ts, mid_price=s, bid_quote=bid, ask_quote=ask,
            spread_quoted=ask - bid, reservation=s,
            inventory=self.inventory,
            pnl_mark=self.mark_to_market(s),
            pnl_spread=self.pnl_spread,
            pnl_costs=self.total_costs,
            cash=self.cash
        )
        self._snapshots.append(snap)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(s) for s in self._snapshots])

# ══════════════════════════════════════════════════════════════════════════════
# 5. MOTOR DE SIMULACIÓN EVENT-DRIVEN
# ══════════════════════════════════════════════════════════════════════════════

class SimulationEngine:
    """Motor de simulación tick-by-tick (resolución 1 segundo)."""

    def __init__(self, mm, rng: np.random.Generator,
                 n_steps: int = N_STEPS, dt: float = DT,
                 kappa_arrival: float = KAPPA_ARRIVAL,
                 kappa_mo: float = KAPPA_MO,
                 hawkes_alpha: float = HAWKES_ALPHA,
                 hawkes_beta: float = HAWKES_BETA,
                 sigma: float = SIGMA_S, S0: float = S0, 
                 T_horizon: float = T_DEFAULT,
                 record_every: int = 60):
        self.mm            = mm
        self.rng           = rng
        self.n_steps       = n_steps
        self.dt            = dt
        self.kappa_arrival = kappa_arrival
        self.kappa_mo      = kappa_mo
        self.hawkes_alpha  = hawkes_alpha
        self.hawkes_beta   = hawkes_beta
        self.sigma         = sigma
        self.S0_sim        = S0
        self.T_horizon     = T_horizon
        self.record_every  = record_every
        self.book          = OrderBook()
        self._metrics: dict = {}

        if hawkes_alpha / hawkes_beta >= 1.0:
            raise ValueError("Hawkes inestable: α/β >= 1.")

    def _generate_price_path(self) -> np.ndarray:
        dW   = self.rng.normal(0, math.sqrt(self.dt), self.n_steps)
        logs = -0.5 * self.sigma**2 * self.dt + self.sigma * dW
        path = self.S0_sim * np.exp(np.cumsum(logs))
        return np.insert(path[:-1], 0, self.S0_sim)

    def _ushape_intensity(self, t_rel: float) -> float:
        factor = 1.2 * (4 * (t_rel - 0.5)**2) + 0.6
        return float(factor)

    def run(self, verbose: bool = True) -> pd.DataFrame:
        t0 = time.time()
        prices = self._generate_price_path()

        bid_oid = ask_oid = None
        hawkes_comp = 0.0
        decay = math.exp(-self.hawkes_beta * self.dt)

        for step in range(self.n_steps):
            s     = prices[step]
            t_abs = step * self.dt
            t_norm= t_abs / (self.n_steps * self.dt)
            t_mm  = t_norm * self.T_horizon

            if bid_oid is not None: self.book.cancel(bid_oid)
            if ask_oid is not None: self.book.cancel(ask_oid)

            bid_p, ask_p = self.mm.quotes(s, t_mm)

            bid_oid = self.book.place_limit('bid', bid_p, LOT_SIZE, t_abs)
            ask_oid = self.book.place_limit('ask', ask_p, LOT_SIZE, t_abs)

            # ── Componente Hawkes ─────────────────────────────────────────
            hawkes_comp *= decay
            lam_base = self.kappa_arrival * self._ushape_intensity(t_norm)
            lam = lam_base + hawkes_comp
            n_mo = self.rng.poisson(lam * self.dt)

            # ── Lógica de Órdenes de Mercado (Exponential Match) ───────────
            for _ in range(n_mo):
                dist = self.rng.exponential(1.0 / self.kappa_mo)
                mo_side = 'buy' if self.rng.random() < 0.5 else 'sell'

                if mo_side == 'buy':
                    if s + dist >= ask_p:
                        # Se pasa el mid-price actual (s) para medir el edge real
                        self.mm.on_fill('ask', ask_p, LOT_SIZE, s) 
                        ask_oid = None
                        hawkes_comp += self.hawkes_alpha
                else:
                    if s - dist <= bid_p:
                        self.mm.on_fill('bid', bid_p, LOT_SIZE, s) 
                        bid_oid = None
                        hawkes_comp += self.hawkes_alpha

            if step % self.record_every == 0:
                self.mm.record_snapshot(t_abs, s, t_mm)

        s_final = prices[-1]
        residual_lots = self.mm.inventory
        if residual_lots != 0:
            residual_cash  = residual_lots * LOT_SIZE * s_final
            self.mm.cash  += residual_cash
            self.mm.inventory = 0
        self.mm.record_snapshot(self.n_steps * self.dt, s_final, self.T_horizon)

        df = self.mm.to_dataframe()
        self._compute_metrics(df, s_final)

        if verbose:
            elapsed = time.time() - t0
            self._print_metrics(elapsed)

        return df

    def _compute_metrics(self, df: pd.DataFrame, s_final: float) -> None:
        pnl_series = df['pnl_mark'].dropna()

        pnl_final = float(pnl_series.iloc[-1])
        pnl_gross = float(df['pnl_spread'].iloc[-1])
        total_costs = float(df['pnl_costs'].iloc[-1])
        pnl_net = pnl_gross - total_costs

        # Sharpe por bloques para suavizar correlación de corto plazo
        BLOCK_MIN   = 30
        SNAP_PER_BLK = BLOCK_MIN
        n_blocks     = len(pnl_series) // SNAP_PER_BLK

        if n_blocks >= 2:
            idx = pnl_series.reset_index(drop=True)
            block_pnls = np.array([
                idx.iloc[min((i + 1) * SNAP_PER_BLK - 1, len(idx) - 1)]
                - idx.iloc[i * SNAP_PER_BLK]
                for i in range(n_blocks)
            ])
            sharpe_d = (block_pnls.mean() / block_pnls.std() * math.sqrt(n_blocks)
                        if block_pnls.std() > 0 else 0.0)
        else:
            sharpe_d = 0.0

        sharpe_a = sharpe_d * math.sqrt(252)

        inv_series = df['inventory'].abs()
        inv_max_dd = int(inv_series.max())

        spread_cap  = float(df['spread_quoted'].mean())
        n_fills_bid = self.mm._n_fills_bid
        n_fills_ask = self.mm._n_fills_ask

        self._metrics = {
            'pnl_gross'   : round(pnl_gross, 2),
            'pnl_costs'   : round(total_costs, 2),
            'pnl_net'     : round(pnl_net, 2),
            'pnl_final'   : round(pnl_final, 2),
            'sharpe_daily': round(sharpe_d, 4),
            'sharpe_annual': round(sharpe_a, 4),
            'max_inv_drawdown_lots': inv_max_dd,
            'spread_mean_captured' : round(spread_cap, 5),
            'n_fills_bid' : n_fills_bid,
            'n_fills_ask' : n_fills_ask,
            'n_fills_total': n_fills_bid + n_fills_ask,
        }

    def _print_metrics(self, elapsed: float) -> None:
        m = self._metrics
        name = type(self.mm).__name__
        print(f'\n{"─"*50}')
        print(f'  RESULTADOS — {name}')
        print(f'{"─"*50}')
        print(f'  P&L bruto (spread)    : ${m["pnl_gross"]:>10.2f}')
        print(f'  Costos transacción    : ${m["pnl_costs"]:>10.2f}')
        print(f'  P&L neto (bruto-cost) : ${m["pnl_net"]:>10.2f}')
        print(f'  P&L mark-to-market    : ${m["pnl_final"]:>10.2f}')
        print(f'  Sharpe diario         : {m["sharpe_daily"]:>10.4f}')
        print(f'  Sharpe anual          : {m["sharpe_annual"]:>10.4f}')
        print(f'  Max inventario (lotes): {m["max_inv_drawdown_lots"]:>10d}')
        print(f'  Spread medio capturado: ${m["spread_mean_captured"]:>10.5f}')
        print(f'  Fills bid / ask       : {m["n_fills_bid"]:>5} / {m["n_fills_ask"]:>5}')
        print(f'  Tiempo simulación     : {elapsed:.2f}s')

    @property
    def metrics(self) -> dict:
        return self._metrics

# ══════════════════════════════════════════════════════════════════════════════
# 6. ANALIZADOR DE SENSIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════

class SensitivityAnalyzer:
    GAMMAS = [0.01, 0.05, 0.10, 0.30, 0.50, 1.00]
    T_HORIZONS = [0.25, 0.50, 1.00]

    def __init__(self, base_kappa_arrival=KAPPA_ARRIVAL, n_steps=N_STEPS):
        self.kappa_arrival = base_kappa_arrival
        self.n_steps = n_steps
        self.results: List[dict] = []

    def run(self) -> pd.DataFrame:
        total = len(self.GAMMAS) * len(self.T_HORIZONS)
        done  = 0
        print(f'\nAnálisis de sensibilidad: {total} combinaciones γ × T')
        print('─' * 60)

        for gamma in self.GAMMAS:
            for T_h in self.T_HORIZONS:
                rng = np.random.default_rng(RNG_SEED)
                mm  = AvellanedaStoikovMM(gamma=gamma, kappa=KAPPA_MO,
                                          T=T_h, sigma=SIGMA_ABS)
                eng = SimulationEngine(mm, rng, n_steps=self.n_steps,
                                       T_horizon=T_h, kappa_arrival=self.kappa_arrival,
                                       kappa_mo=KAPPA_MO)
                eng.run(verbose=False)
                m = eng.metrics
                self.results.append({
                    'gamma'         : gamma,
                    'T_horizon'     : T_h,
                    'pnl_final'     : m['pnl_final'],
                    'sharpe_daily'  : m['sharpe_daily'],
                    'sharpe_annual' : m['sharpe_annual'],
                    'max_inv_dd'    : m['max_inv_drawdown_lots'],
                    'spread_cap'    : m['spread_mean_captured'],
                    'n_trades'      : m['n_fills_total'],
                })
                done += 1
                print(f'  [{done:2d}/{total}] γ={gamma:.2f}  T={T_h:.2f} → '
                      f'P&L=${m["pnl_final"]:7.2f}  Sharpe={m["sharpe_daily"]:.4f}  '
                      f'MaxInv={m["max_inv_drawdown_lots"]}lots')

        return pd.DataFrame(self.results)

# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

def plot_simulation(df_as: pd.DataFrame, df_naive: pd.DataFrame,
                    metrics_as: dict, metrics_naive: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Simulación Market Making: Avellaneda-Stoikov vs Naive',
                 fontsize=13, y=1.01)

    ax = axes[0, 0]
    ax.plot(df_as['ts'] / 3600, df_as['pnl_mark'],
            color=COLORS['teal'], lw=1.2, label=f'A-S (P&L=${metrics_as["pnl_final"]:.2f})')
    ax.plot(df_naive['ts'] / 3600, df_naive['pnl_mark'],
            color=COLORS['red'], lw=1.2, ls='--',
            label=f'Naive (P&L=${metrics_naive["pnl_final"]:.2f})')
    ax.axhline(0, color=COLORS['muted'], lw=0.5)
    ax.set_xlabel('Hora de sesión (h)')
    ax.set_ylabel('P&L mark-to-market ($)')
    ax.set_title('P&L a lo largo del día')
    ax.legend(fontsize=9)
    ax.grid(True)

    ax = axes[0, 1]
    ax.fill_between(df_as['ts'] / 3600, df_as['inventory'],
                    color=COLORS['teal'], alpha=0.4, label='A-S')
    ax.fill_between(df_naive['ts'] / 3600, df_naive['inventory'],
                    color=COLORS['red'], alpha=0.3, label='Naive')
    ax.axhline(0, color='white', lw=0.5)
    ax.axhline(Q_MAX, color=COLORS['gold'], lw=0.8, ls='--', label=f'±{Q_MAX} lotes')
    ax.axhline(-Q_MAX, color=COLORS['gold'], lw=0.8, ls='--')
    ax.set_xlabel('Hora de sesión (h)')
    ax.set_ylabel('Inventario (lotes)')
    ax.set_title('Inventario a lo largo del día')
    ax.legend(fontsize=9)
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(df_as['ts'] / 3600, df_as['mid_price'],
            color='white', lw=0.8, alpha=0.6, label='Mid price')
    ax.plot(df_as['ts'] / 3600, df_as['reservation'],
            color=COLORS['purple'], lw=1.0, label='Reservation price (A-S)')
    ax.plot(df_as['ts'] / 3600, df_as['bid_quote'],
            color=COLORS['teal'], lw=0.6, alpha=0.7, label='Bid quote')
    ax.plot(df_as['ts'] / 3600, df_as['ask_quote'],
            color=COLORS['red'], lw=0.6, alpha=0.7, label='Ask quote')
    ax.set_xlabel('Hora de sesión (h)')
    ax.set_ylabel('Precio ($)')
    ax.set_title('Quotes A-S: Mid, Reservation, Bid*, Ask*')
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(df_as['ts'] / 3600, df_as['spread_quoted'] * 100,
            color=COLORS['blue'], lw=0.8, label=f'A-S spread (¢)')
    ax.plot(df_naive['ts'] / 3600, df_naive['spread_quoted'] * 100,
            color=COLORS['gold'], lw=0.8, ls='--', label=f'Naive spread (¢)')
    ax.set_xlabel('Hora de sesión (h)')
    ax.set_ylabel('Spread (¢)')
    ax.set_title(f'Spread cotizado | A-S Sharpe={metrics_as["sharpe_daily"]:.3f}')
    ax.legend(fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    plt.show() 


def plot_sensitivity(df_sens: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Análisis de Sensibilidad γ × T', fontsize=12)

    metrics_to_plot = [
        ('pnl_final',   'P&L final ($)',      COLORS['teal']),
        ('sharpe_daily','Sharpe diario',       COLORS['blue']),
        ('max_inv_dd',  'Max inventario (lots)', COLORS['red']),
    ]
    gammas = sorted(df_sens['gamma'].unique())
    Ts     = sorted(df_sens['T_horizon'].unique())

    for ax, (col, label, color) in zip(axes, metrics_to_plot):
        matrix = df_sens.pivot(index='gamma', columns='T_horizon', values=col)
        data = matrix.values
        im = ax.imshow(data, cmap='RdYlGn' if col != 'max_inv_dd' else 'RdYlGn_r',
                       aspect='auto')
        ax.set_xticks(range(len(Ts)));     ax.set_xticklabels([f'T={t}' for t in Ts])
        ax.set_yticks(range(len(gammas))); ax.set_yticklabels([f'γ={g}' for g in gammas])
        for i in range(len(gammas)):
            for j in range(len(Ts)):
                ax.text(j, i, f'{data[i,j]:.2f}',
                        ha='center', va='center', fontsize=8, color='white')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(label)
        ax.set_xlabel('Horizonte T')
        ax.set_ylabel('Aversión γ')

    plt.tight_layout()
    plt.show() 

# ══════════════════════════════════════════════════════════════════════════════
# 8. TESTS UNITARIOS (18 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestOrderBook(unittest.TestCase):
    def setUp(self):
        self.book = OrderBook()

    def test_best_bid_after_single_order(self):
        self.book.place_limit('bid', 99.90, 100, 0.0)
        self.assertAlmostEqual(self.book.best_bid(), 99.90)

    def test_best_ask_after_single_order(self):
        self.book.place_limit('ask', 100.10, 100, 0.0)
        self.assertAlmostEqual(self.book.best_ask(), 100.10)

    def test_cancel_removes_from_top(self):
        oid = self.book.place_limit('bid', 99.90, 100, 0.0)
        self.book.place_limit('bid', 99.80, 100, 0.0)
        self.book.cancel(oid)
        self.assertAlmostEqual(self.book.best_bid(), 99.80)

    def test_mid_price(self):
        self.book.place_limit('bid', 99.90, 100, 0.0)
        self.book.place_limit('ask', 100.10, 100, 0.0)
        self.assertAlmostEqual(self.book.mid_price(), 100.00)

    def test_spread(self):
        self.book.place_limit('bid', 99.90, 100, 0.0)
        self.book.place_limit('ask', 100.10, 100, 0.0)
        self.assertAlmostEqual(self.book.spread(), 0.20, places=5)

    def test_empty_book_returns_none(self):
        self.assertIsNone(self.book.best_bid())
        self.assertIsNone(self.book.best_ask())
        self.assertIsNone(self.book.mid_price())

class TestAvellanedaStoikov(unittest.TestCase):
    def setUp(self):
        self.mm = AvellanedaStoikovMM(gamma=0.1, kappa=1.5, sigma=SIGMA_ABS,
                                      T=1.0, q_max=8)

    def test_reservation_price_zero_inventory(self):
        r = self.mm.reservation_price(100.0, 0, 0.0)
        self.assertAlmostEqual(r, 100.0, places=4)

    def test_reservation_price_positive_inventory(self):
        self.mm.inventory = 3
        r = self.mm.reservation_price(100.0, 3, 0.5)
        self.assertLess(r, 100.0)

    def test_reservation_price_negative_inventory(self):
        r = self.mm.reservation_price(100.0, -3, 0.5)
        self.assertGreater(r, 100.0)

    def test_optimal_spread_positive(self):
        d = self.mm.optimal_spread(0.5)
        self.assertGreater(d, 0.0)

    def test_optimal_spread_increases_near_T(self):
        d_early = self.mm.optimal_spread(0.01)
        d_late  = self.mm.optimal_spread(0.99)
        self.assertGreaterEqual(d_early, d_late)

    def test_quotes_ask_greater_than_bid(self):
        bid, ask = self.mm.quotes(100.0, 0.5)
        self.assertGreater(ask - bid, TICK / 2)

    def test_quotes_symmetric_zero_inventory(self):
        self.mm.inventory = 0
        bid, ask = self.mm.quotes(100.0, 0.5)
        self.assertAlmostEqual((bid + ask) / 2, 100.0, places=0)

    def test_mark_to_market_zero_inventory(self):
        self.mm.inventory = 0
        self.mm.cash = 0.0
        self.assertAlmostEqual(self.mm.mark_to_market(100.0), 0.0)

    def test_on_fill_bid_increases_inventory(self):
        initial_inv = self.mm.inventory
        self.mm.on_fill('bid', 99.9, LOT_SIZE, mid_price=100.0)
        self.assertEqual(self.mm.inventory, initial_inv + 1)

    def test_on_fill_ask_decreases_inventory(self):
        self.mm.inventory = 2
        self.mm.on_fill('ask', 100.1, LOT_SIZE, mid_price=100.0)
        self.assertEqual(self.mm.inventory, 1)

class TestSimulationEngine(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(RNG_SEED)
        self.mm  = AvellanedaStoikovMM(gamma=0.1, kappa=1.5, sigma=SIGMA_ABS, T=1.0)
        self.eng = SimulationEngine(self.mm, self.rng, n_steps=1000,
                                    T_horizon=1.0, record_every=100)

    def test_simulation_runs_without_error(self):
        df = self.eng.run(verbose=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_metrics_computed(self):
        self.eng.run(verbose=False)
        self.assertIn('pnl_final', self.eng.metrics)
        self.assertIn('sharpe_daily', self.eng.metrics)

    def test_final_inventory_zero_after_close(self):
        self.eng.run(verbose=False)
        self.assertEqual(self.mm.inventory, 0)

    def test_pnl_is_finite(self):
        self.eng.run(verbose=False)
        self.assertTrue(math.isfinite(self.eng.metrics['pnl_final']))

# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main_simulation() -> None:
    print('╔' + '═'*58 + '╗')
    print('║  Simulación Market Making — Avellaneda-Stoikov (2008)    ║')
    print('╚' + '═'*58 + '╝')

    rng_as    = np.random.default_rng(RNG_SEED)
    rng_naive = np.random.default_rng(RNG_SEED)

    print('\n[1/2] Simulando Avellaneda-Stoikov MM...')
    mm_as  = AvellanedaStoikovMM(gamma=GAMMA_DEFAULT, kappa=KAPPA_MO,
                                  sigma=SIGMA_ABS, T=T_DEFAULT, q_max=Q_MAX)
    eng_as = SimulationEngine(mm_as, rng_as, n_steps=N_STEPS,
                               T_horizon=T_DEFAULT, kappa_arrival=KAPPA_ARRIVAL,
                               kappa_mo=KAPPA_MO)
    df_as  = eng_as.run(verbose=True)

    print('\n[2/2] Simulando Naive MM (benchmark)...')
    mm_naive  = NaiveMM(q_max=Q_MAX)
    eng_naive = SimulationEngine(mm_naive, rng_naive, n_steps=N_STEPS,
                                  T_horizon=T_DEFAULT, kappa_arrival=KAPPA_ARRIVAL,
                                  kappa_mo=KAPPA_MO)
    df_naive  = eng_naive.run(verbose=True)

    print('\n' + '═'*58)
    print('  COMPARATIVA A-S vs NAIVE')
    print('═'*58)
    metrics_labels = {
        'pnl_gross'           : 'P&L bruto — spread ($)',
        'pnl_costs'           : 'Costos transacción ($)',
        'pnl_net'             : 'P&L neto bruto−costos ($)',
        'pnl_final'           : 'P&L mark-to-market ($)',
        'sharpe_daily'        : 'Sharpe diario',
        'sharpe_annual'       : 'Sharpe anual',
        'max_inv_drawdown_lots': 'Max inventario (lotes)',
        'spread_mean_captured': 'Spread medio capturado ($)',
        'n_fills_total'       : 'Nº fills total',
    }
    for key, label in metrics_labels.items():
        v_as    = eng_as.metrics.get(key, 'N/A')
        v_naive = eng_naive.metrics.get(key, 'N/A')
        try:
            diff = f'+{v_as - v_naive:.4g}' if isinstance(v_as, float) else ''
        except Exception:
            diff = ''
        print(f'  {label:<34}: A-S={v_as}  Naive={v_naive}  Δ={diff}')

    print('\nGenerando gráficas...')
    plot_simulation(df_as, df_naive, eng_as.metrics, eng_naive.metrics)


def main_sensitivity() -> None:
    analyzer = SensitivityAnalyzer(base_kappa_arrival=KAPPA_ARRIVAL, n_steps=N_STEPS)
    df_sens  = analyzer.run()

    print('\n' + '═'*60)
    print('  TABLA DE SENSIBILIDAD (P&L)')
    print('═'*60)
    pivot = df_sens.pivot(index='gamma', columns='T_horizon', values='pnl_final')
    print(pivot.to_string())

    plot_sensitivity(df_sens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulador Market Making A-S')
    parser.add_argument('--test',        action='store_true', help='Ejecutar tests unitarios')
    parser.add_argument('--sensitivity', action='store_true', help='Análisis de sensibilidad')
    args = parser.parse_args()

    if args.test:
        print('Ejecutando tests unitarios...\n')
        loader = unittest.TestLoader()
        suite  = unittest.TestSuite()
        for cls in [TestOrderBook, TestAvellanedaStoikov, TestSimulationEngine]:
            suite.addTests(loader.loadTestsFromTestCase(cls))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        print(f'\n{"PASS" if result.wasSuccessful() else "FAIL"} — '
              f'{result.testsRun} tests, {len(result.failures)} fallos, '
              f'{len(result.errors)} errores')
    elif args.sensitivity:
        main_sensitivity()
    else:
        main_simulation()