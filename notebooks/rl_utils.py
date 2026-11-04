"""
rl_utils.py — Utilidades para la clase de Aprendizaje por Refuerzo (Duoc UC).

Se usa en el notebook 02 (Cliff Walking, Q-learning, SARSA y las métricas)
y en los ejercicios 4 y 6 del notebook 04.

Contiene:
  * CliffWalking: entorno tabular clásico (Sutton & Barto, ej. 6.6), sin dependencias externas.
  * Agentes tabulares: aleatorio, Q-learning, SARSA.
  * Funciones de métricas para analizar curvas de aprendizaje.

Solo usa numpy y pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Entorno: CliffWalking 4x12
# ---------------------------------------------------------------------------
ACCIONES = {0: "arriba", 1: "derecha", 2: "abajo", 3: "izquierda"}
_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


class CliffWalking:
    """Grilla 4x12. Inicio abajo-izquierda, meta abajo-derecha, acantilado entre ambos.
    Recompensa -1 por paso, -100 al caer al acantilado (vuelve al inicio).
    El episodio se corta en `max_pasos` para evitar loops infinitos."""

    n_filas, n_cols = 4, 12
    n_estados = n_filas * n_cols
    n_acciones = 4

    def __init__(self, max_pasos: int = 200, seed: int | None = None):
        self.max_pasos = max_pasos
        self.rng = np.random.default_rng(seed)
        self.inicio = (3, 0)
        self.meta = (3, 11)

    def _id(self, pos):
        return pos[0] * self.n_cols + pos[1]

    def reset(self):
        self.pos = self.inicio
        self.t = 0
        return self._id(self.pos)

    def step(self, a: int):
        df, dc = _DELTAS[a]
        f = min(max(self.pos[0] + df, 0), self.n_filas - 1)
        c = min(max(self.pos[1] + dc, 0), self.n_cols - 1)
        self.pos = (f, c)
        self.t += 1
        cayo = f == 3 and 1 <= c <= 10
        if cayo:
            self.pos = self.inicio
            r = -100.0
        else:
            r = -1.0
        llego = self.pos == self.meta
        truncado = self.t >= self.max_pasos
        info = {"cayo": cayo, "llego": llego}
        return self._id(self.pos), r, llego or truncado, info


# ---------------------------------------------------------------------------
# Agentes tabulares
# ---------------------------------------------------------------------------
def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1])), True
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max()))), False


def entrenar(
    algoritmo: str,
    n_episodios: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    eps_inicial: float = 0.3,
    eps_final: float = 0.01,
    decaimiento: bool = True,
    seed: int = 0,
    env=None,
):
    """Entrena un agente y devuelve (Q, DataFrame con log por episodio).

    algoritmo: 'aleatorio' | 'q_learning' | 'sarsa'
    """
    rng = np.random.default_rng(seed)
    env = env or CliffWalking(seed=seed)
    Q = np.zeros((env.n_estados, env.n_acciones))
    filas = []
    for ep in range(n_episodios):
        if decaimiento:
            eps = max(eps_final, eps_inicial * (1 - ep / (0.8 * n_episodios)))
        else:
            eps = eps_inicial
        if algoritmo == "aleatorio":
            eps = 1.0
        s = env.reset()
        a, expl = epsilon_greedy(Q, s, eps, rng)
        G, largo, n_expl, caidas, td_abs = 0.0, 0, 0, 0, []
        done = False
        while not done:
            s2, r, done, info = env.step(a)
            a2, expl2 = epsilon_greedy(Q, s2, eps, rng)
            if algoritmo == "q_learning":
                objetivo = r + (0 if info["llego"] else gamma * Q[s2].max())
            elif algoritmo == "sarsa":
                objetivo = r + (0 if info["llego"] else gamma * Q[s2, a2])
            else:
                objetivo = Q[s, a]
            td = objetivo - Q[s, a]
            if algoritmo != "aleatorio":
                Q[s, a] += alpha * td
            td_abs.append(abs(td))
            G += r
            largo += 1
            n_expl += expl
            caidas += info["cayo"]
            s, a, expl = s2, a2, expl2
        filas.append(
            dict(
                algoritmo=algoritmo,
                seed=seed,
                episodio=ep,
                retorno=G,
                largo=largo,
                exito=int(info["llego"]),
                caidas=caidas,
                epsilon=round(eps, 4),
                frac_exploracion=n_expl / largo,
                td_error_medio=float(np.mean(td_abs)),
            )
        )
    return Q, pd.DataFrame(filas)


def evaluar_greedy(Q, n_episodios: int = 20, seed: int = 123, env=None):
    """Ejecuta la política greedy (eps=0) SIN aprender. Devuelve retornos."""
    rng = np.random.default_rng(seed)
    env = env or CliffWalking(seed=seed)
    retornos = []
    for _ in range(n_episodios):
        s, done, G = env.reset(), False, 0.0
        while not done:
            a, _ = epsilon_greedy(Q, s, 0.0, rng)
            s, r, done, _ = env.step(a)
            G += r
        retornos.append(G)
    return np.array(retornos)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def media_movil(x, ventana: int = 20):
    return pd.Series(x).rolling(ventana, min_periods=1).mean().to_numpy()


def curva_con_ic(df, col="retorno", grupo="algoritmo", ventana=20, z=1.96):
    """Media entre semillas e intervalo de confianza 95% (aprox. normal) por episodio,
    sobre la media móvil de cada semilla."""
    df = df.copy()
    df["suav"] = df.groupby([grupo, "seed"])[col].transform(
        lambda s: s.rolling(ventana, min_periods=1).mean()
    )
    agg = df.groupby([grupo, "episodio"])["suav"].agg(["mean", "std", "count"]).reset_index()
    agg["ic"] = z * agg["std"] / np.sqrt(agg["count"])
    return agg


def auc_normalizada(retornos, r_min, r_max):
    """Área bajo la curva de aprendizaje normalizada a [0,1]. Mide eficiencia muestral."""
    r = (np.asarray(retornos) - r_min) / (r_max - r_min)
    return float(np.clip(r, 0, 1).mean())


def episodios_hasta_umbral(retornos, umbral, ventana=20):
    """Primer episodio donde la media móvil supera el umbral (None si nunca)."""
    mm = media_movil(retornos, ventana)
    idx = np.flatnonzero(mm >= umbral)
    return int(idx[0]) if len(idx) else None


def regret_acumulado(retornos, retorno_optimo):
    """Suma de lo que se 'dejó de ganar' respecto al óptimo en cada episodio."""
    return np.cumsum(retorno_optimo - np.asarray(retornos))


def resumen_metricas(df, r_min=-200, r_max=-13, umbral=-30, ultimos=50, retorno_optimo=-13):
    filas = []
    for (alg, seed), g in df.groupby(["algoritmo", "seed"]):
        r = g.sort_values("episodio")["retorno"].to_numpy()
        filas.append(
            dict(
                algoritmo=alg,
                seed=seed,
                retorno_final=r[-ultimos:].mean(),
                std_final=r[-ultimos:].std(),
                exito_final=g["exito"].to_numpy()[-ultimos:].mean(),
                auc=auc_normalizada(r, r_min, r_max),
                ep_hasta_umbral=episodios_hasta_umbral(r, umbral),
                regret_total=regret_acumulado(r, retorno_optimo)[-1],
                caidas_totales=g["caidas"].sum(),
            )
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Entorno 2: Bodega de inventario (para el notebook de técnicas resueltas)
# ---------------------------------------------------------------------------
class Inventario:
    """Control de inventario de una ferretería durante 30 días.

    Observación (tupla de enteros): (dia, stock, demanda_ayer)
      - stock en unidades 0..20, demanda diaria Poisson con media que cambia por día de semana.
    Acciones: pedir 0, 3, 6 o 10 unidades (llegan al día siguiente).
    Recompensa diaria: ventas*precio - pedido*costo - stock*almacenaje - quiebre*penalización.
    El estado crudo tiene 30*21*15 = 9450 combinaciones -> conviene agregar estados.
    """

    PEDIDOS = [0, 3, 6, 10]
    n_acciones = 4

    def __init__(self, T=30, precio=5.0, costo=2.0, almacenaje=0.3, quiebre=4.0,
                 media_demanda=(3, 3, 4, 4, 6, 8, 2), seed=None):
        self.T, self.precio, self.costo = T, precio, costo
        self.almacenaje, self.quiebre = almacenaje, quiebre
        self.media = np.array(media_demanda, dtype=float)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.dia, self.stock, self.en_camino, self.dem_ayer = 0, 8, 0, 3
        return (self.dia, self.stock, self.dem_ayer)

    def step(self, a):
        self.stock = min(20, self.stock + self.en_camino)
        pedido = self.PEDIDOS[a]
        d = int(self.rng.poisson(self.media[self.dia % 7]))
        ventas = min(d, self.stock)
        faltante = d - ventas
        self.stock -= ventas
        r = ventas * self.precio - pedido * self.costo - self.stock * self.almacenaje - faltante * self.quiebre
        self.en_camino = pedido
        self.dem_ayer = min(d, 14)
        self.dia += 1
        done = self.dia >= self.T
        return (self.dia, self.stock, self.dem_ayer), float(r), done, {"faltante": faltante}


# ---------------------------------------------------------------------------
# Entorno 3: Venta de una camioneta (problema de PARADA ÓPTIMA)
# ---------------------------------------------------------------------------
class VentaCamioneta:
    """Cada día llega una oferta por tu camioneta. Puedes esperar (0) o aceptar (1).
    Esperar cuesta 0.15 (M$) por día (estacionamiento, aviso). Si llegas al día 30 vendes a la
    automotora por 6.0. Las ofertas son ruidosas; la media sube hasta el día 15 y luego baja.

    Observación: (dia, oferta_bin 0..9). Recompensa: -0.15 por día de espera; oferta al aceptar.
    Es la misma estructura que muchos problemas reales: recompensa grande y tardía
    + una acción que TERMINA el episodio."""

    n_acciones = 2

    def __init__(self, T=30, seed=None):
        self.T = T
        self.rng = np.random.default_rng(seed)

    def _media(self, d):
        return 8.0 + 2.0 * np.exp(-((d - 15) / 6) ** 2)

    def reset(self):
        self.dia = 0
        self.oferta = self.rng.normal(self._media(0), 1.2)
        return (self.dia, self._bin())

    def _bin(self):
        return int(np.clip((self.oferta - 5) // 1, 0, 9))

    def step(self, a):
        if a == 1:
            return (self.dia, self._bin()), float(self.oferta), True, {"vendio": self.dia}
        self.dia += 1
        if self.dia >= self.T:
            return (self.dia - 1, self._bin()), -0.15 + 6.0, True, {"vendio": self.T}
        self.oferta = self.rng.normal(self._media(self.dia), 1.2)
        return (self.dia, self._bin()), -0.15, False, {"vendio": None}
