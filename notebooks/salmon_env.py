"""
salmon_env.py — Entorno del desafío (notebook 05): "Centro de cultivo de salmón — Seno de Reloncaví".

Lo que el agente ve, las acciones y la recompensa están también en la primera
celda de ese notebook. Este archivo es la referencia cuando haga falta el detalle.

Cada grupo recibe un entorno con PARÁMETROS PROPIOS (derivados de su código de grupo).
Por eso la política óptima de un grupo no sirve para otro: hay que experimentar en el
entorno propio, medir e iterar.

Interfaz (estilo Gymnasium, sin dependencias):
    env = SalmonFarmEnv(grupo="G07", seed=0)
    obs = env.reset()
    obs, recompensa, terminado, info = env.step(accion)

OBSERVACIÓN (tupla de 4 enteros) — lo único que ve el agente:
    0. dia               0..59
    1. biomasa_bin       0..9   (biomasa en tramos de 10 t desde 80 t)
    2. o2_sensor         0..4   (oxígeno disuelto; el sensor reporta con 1 día de RETRASO y a veces falla ±1)
    3. tendencia_precio  0=baja, 1=estable, 2=sube (señal de mercado a 3 días, con ruido)

ACCIONES:
    0 = ayuno (no alimentar)
    1 = ración normal
    2 = ración alta
    3 = aireación + ración normal (cuesta extra, sube el oxígeno)
    4 = COSECHAR (termina el episodio, vende la biomasa al precio del día)

RECOMPENSA (miles de USD):
    - Diaria: -costo de alimento -costo de aireación -multa sanitaria si hay mortalidad.
    - Al cosechar: + biomasa * precio(día).
    - Si llega al día 60 sin cosechar: cosecha forzada al 70% del precio.

Hay variables OCULTAS (el agente no las ve): oxígeno real, si hay un bloom de algas activo,
biomasa exacta y el precio exacto. Es un problema parcialmente observable (POMDP).

PRESUPUESTO: cada agente se entrena con a lo más PASOS_POR_CORRIDA interacciones, y el grupo dispone
de PRESUPUESTO_PASOS interacciones en total (todas las corridas, semillas y pruebas, sumando todas las
instancias creadas en la sesión). Deben reportar env.pasos_usados().
"""
from __future__ import annotations

import hashlib
import numpy as np

T = 60
N_ACCIONES = 5
NOMBRES_ACCIONES = ["ayuno", "normal", "alta", "aireacion+normal", "cosechar"]
PRESUPUESTO_PASOS = 1_500_000   # total de la sesión (todas las corridas)
PASOS_POR_CORRIDA = 150_000     # máximo para entrenar UN agente

_contador_pasos: dict[str, int] = {}


def parametros_grupo(grupo: str) -> dict:
    """Parámetros del entorno para un grupo. Estudiantes: NO es necesario (ni recomendable)
    leerlos; el desafío es descubrir el comportamiento experimentando. En la defensa se pregunta
    qué descubrieron y cómo lo midieron."""
    h = hashlib.sha256(("reloncavi-" + grupo.strip().upper()).encode()).digest()
    u = [b / 255 for b in h]  # 32 números en [0,1]

    def rango(i, lo, hi):
        return lo + (hi - lo) * u[i]

    return dict(
        crecimiento=rango(0, 0.012, 0.018),   # crecimiento diario con ración normal
        bloom_p=rango(1, 0.03, 0.08),         # prob. diaria de inicio de bloom de algas
        bloom_largo=int(round(rango(2, 3, 7))),
        bloom_caida_o2=1 + int(u[3] > 0.5),   # cuánto baja el O2 un bloom
        mortalidad=rango(4, 0.04, 0.10),      # fracción de biomasa perdida por día con O2 crítico
        pico_precio=int(round(rango(5, 32, 52))),  # pico medio; cada temporada se mueve ±6 días
        ancho_precio=rango(6, 6, 12),
        amp_precio=rango(7, 0.25, 0.60),
        carga_max=rango(8, 140, 170),         # sobre esta biomasa el O2 cae 1 nivel
    )


class SalmonFarmEnv:
    COEF_ALIMENTO = [0.0, 0.045, 0.090, 0.045]
    MULT_CRECIMIENTO = [0.0, 1.0, 1.5, 1.0]
    COSTO_AIREACION = 2.5
    MULTA_MORTALIDAD = 3.0
    PRECIO_BASE = 5.0

    def __init__(self, grupo: str = "DEMO", seed: int | None = None, params: dict | None = None,
                 contar_presupuesto: bool = True):
        self.grupo = grupo.strip().upper()
        self.p = params if params is not None else parametros_grupo(self.grupo)
        self.rng = np.random.default_rng(seed)
        self.contar = contar_presupuesto
        _contador_pasos.setdefault(self.grupo, 0)

    # ---------------------------------------------------------------- utilidades
    def pasos_usados(self) -> int:
        return _contador_pasos[self.grupo]

    def _precio(self, d):
        p = self.p
        return self.PRECIO_BASE * (1 + p["amp_precio"] * np.exp(-((d - self.pico) / p["ancho_precio"]) ** 2))

    def _obs(self):
        bio_bin = int(np.clip((self.biomasa - 80) // 10, 0, 9))
        o2s = self.o2_ayer
        if self.rng.random() < 0.15:
            o2s = int(np.clip(o2s + self.rng.choice([-1, 1]), 0, 4))
        dif = self._precio(min(self.dia + 3, T - 1)) - self._precio(self.dia)
        tend = 2 if dif > 0.05 else (0 if dif < -0.05 else 1)
        if self.rng.random() < 0.2:
            tend = int(self.rng.integers(3))
        return (self.dia, bio_bin, int(o2s), tend)

    # ---------------------------------------------------------------- API
    def reset(self):
        self.dia = 0
        self.biomasa = 100.0 + self.rng.normal(0, 4)
        self.o2 = 4
        self.o2_ayer = 4
        self.bloom_restante = 0
        # el pico de precio de la temporada se desplaza de forma aleatoria cada episodio
        self.pico = self.p["pico_precio"] + int(self.rng.integers(-6, 7))
        return self._obs()

    def step(self, a: int):
        a = int(a)
        if not 0 <= a < N_ACCIONES:
            raise ValueError(f"Acción inválida {a}")
        if self.contar:
            if _contador_pasos[self.grupo] >= PRESUPUESTO_PASOS:
                raise RuntimeError(
                    f"Presupuesto agotado ({PRESUPUESTO_PASOS} pasos) para el grupo {self.grupo}. "
                    "Reinicia el kernel solo si el docente lo autoriza y repórtalo en el informe."
                )
            _contador_pasos[self.grupo] += 1

        p = self.p
        info = {"mortalidad": False}

        if a == 4:  # cosechar
            r = self.biomasa * self._precio(self.dia)
            info.update(cosecha=True, dia_cosecha=self.dia, biomasa=self.biomasa)
            return self._obs(), float(r), True, info

        # --- bloom (oculto) ---
        if self.bloom_restante > 0:
            self.bloom_restante -= 1
        else:
            prob = p["bloom_p"] * (2.0 if 20 <= self.dia <= 45 else 1.0)
            if self.rng.random() < prob:
                self.bloom_restante = p["bloom_largo"]
        bloom = self.bloom_restante > 0

        # --- oxígeno real (oculto) ---
        objetivo = 4 - (p["bloom_caida_o2"] if bloom else 0) - (a == 2) - (self.biomasa > p["carga_max"]) \
            + 2 * (a == 3) + (a == 0)
        objetivo = int(np.clip(objetivo, 0, 4))
        self.o2_ayer = self.o2
        self.o2 += int(np.sign(objetivo - self.o2))
        if self.rng.random() < 0.1:
            self.o2 += int(self.rng.choice([-1, 1]))
        self.o2 = int(np.clip(self.o2, 0, 4))

        # --- crecimiento y mortalidad ---
        f_o2 = 1.0 if self.o2 >= 3 else (0.6 if self.o2 == 2 else 0.2)
        crec = p["crecimiento"] * self.MULT_CRECIMIENTO[a] * f_o2
        costo = self.biomasa * self.COEF_ALIMENTO[a] + (self.COSTO_AIREACION if a == 3 else 0.0)
        self.biomasa *= 1 + crec
        r = -costo
        if self.o2 <= 1:
            frac = p["mortalidad"] * (2.0 if self.o2 == 0 else 1.0) * (1.5 if a in (1, 2) else 1.0)
            self.biomasa *= 1 - frac
            r -= self.MULTA_MORTALIDAD
            info["mortalidad"] = True

        self.dia += 1
        if self.dia >= T:  # cosecha forzada
            r += 0.7 * self.biomasa * self._precio(T - 1)
            info.update(cosecha=False, dia_cosecha=T, biomasa=self.biomasa)
            return self._obs(), float(r), True, info
        return self._obs(), float(r), False, info


# ---------------------------------------------------------------- políticas de referencia
class PoliticaAleatoria:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def actuar(self, obs):
        # cosechar con baja probabilidad para que no termine siempre el día 1
        return int(self.rng.choice(5, p=[0.24, 0.24, 0.24, 0.24, 0.04]))


class PoliticaOperador:
    """Heurística del operador actual del centro (la línea base a superar)."""

    def reset(self):
        pass

    def actuar(self, obs):
        dia, bio, o2, tend = obs
        if dia >= 50:
            return 4
        if o2 <= 1:
            return 3
        return 1


def evaluar_politica(politica, grupo, n_episodios=100, seed=10_000, params=None):
    """Evalúa SIN contar presupuesto (la evaluación no es entrenamiento).
    Devuelve dict con retornos y estadísticas operativas."""
    env = SalmonFarmEnv(grupo, seed=seed, params=params, contar_presupuesto=False)
    retornos, dias, mort = [], [], []
    for _ in range(n_episodios):
        obs, done, G, m = env.reset(), False, 0.0, 0
        if hasattr(politica, "reset"):
            politica.reset()
        while not done:
            obs, r, done, info = env.step(politica.actuar(obs))
            G += r
            m += info["mortalidad"]
        retornos.append(G)
        dias.append(info["dia_cosecha"])
        mort.append(m)
    retornos = np.array(retornos)
    return dict(
        retorno_medio=retornos.mean(),
        ic95=1.96 * retornos.std(ddof=1) / np.sqrt(len(retornos)),
        p10=np.percentile(retornos, 10),
        dia_cosecha_medio=float(np.mean(dias)),
        dias_mortalidad_medio=float(np.mean(mort)),
        retornos=retornos,
    )
