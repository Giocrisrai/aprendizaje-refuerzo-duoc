"""Grilla del héroe: Q-learning que se ve en clase.

Se usa en el notebook 01 (el ejemplo resuelto) y otra vez en el 06,
donde la misma ruta se busca sin tabla Q.

El mapa no cambia, así que el estado de Markov es la casilla (fila, columna).
Fila 0 está arriba. La misma idea que la práctica del diplomado (héroe, trofeo,
zombis), con coordenadas simples y una corrida de menos de un segundo.

Recompensa: trofeo +1000 y termina, zombi -100 y termina, cualquier otro
paso cuesta `costo_paso` (por defecto -1). Chocar con el borde no mueve
al héroe y también cobra el paso.
"""

from __future__ import annotations

import numpy as np

VACIO = "⚪"
ZOMBI = "🧟"
TROFEO = "🏆"
HEROE = "🙃"

ARRIBA, ABAJO, IZQUIERDA, DERECHA = 0, 1, 2, 3
ACCIONES = (ARRIBA, ABAJO, IZQUIERDA, DERECHA)
NOMBRE_ACCION = ("arriba", "abajo", "izquierda", "derecha")
_DELTAS = {ARRIBA: (-1, 0), ABAJO: (1, 0), IZQUIERDA: (0, -1), DERECHA: (0, 1)}

# Trofeo arriba a la izquierda. El zombi de la columna 0 bloquea el atajo.
MAPA_CLASE = (
    (TROFEO, VACIO, VACIO),
    (ZOMBI, VACIO, ZOMBI),
    (VACIO, VACIO, VACIO),
)
INICIO_CLASE = (2, 0)


def recompensa_de(celda: str, costo_paso: float = -1.0) -> tuple[float, bool]:
    """Recompensa y si el episodio termina, según lo que hay en la casilla."""
    if celda == ZOMBI:
        return -100.0, True
    if celda == TROFEO:
        return 1000.0, True
    if celda == VACIO:
        return float(costo_paso), False
    raise ValueError(f"casilla desconocida: {celda!r}")


class Estado:
    def __init__(self, mapa, pos: tuple[int, int]):
        self.mapa = tuple(tuple(fila) for fila in mapa)
        self.pos = (int(pos[0]), int(pos[1]))
        self.alto = len(self.mapa)
        self.ancho = len(self.mapa[0])
        if not (0 <= self.pos[0] < self.alto and 0 <= self.pos[1] < self.ancho):
            raise ValueError(f"posición fuera del mapa: {self.pos}")

    def copia(self) -> "Estado":
        return Estado(self.mapa, self.pos)

    def celda(self, pos: tuple[int, int] | None = None) -> str:
        f, c = self.pos if pos is None else pos
        return self.mapa[f][c]

    def mover(self, accion: int) -> "Estado":
        f, c = self.pos
        df, dc = _DELTAS[int(accion)]
        nf, nc = f + df, c + dc
        if not (0 <= nf < self.alto and 0 <= nc < self.ancho):
            nf, nc = f, c
        return Estado(self.mapa, (nf, nc))

    def texto(self) -> str:
        filas = []
        for i, fila in enumerate(self.mapa):
            celdas = []
            for j, celda in enumerate(fila):
                celdas.append(HEROE if (i, j) == self.pos else celda)
            filas.append(" ".join(celdas))
        return "\n".join(filas)

    def __str__(self) -> str:
        return self.texto()


def inicio_clase() -> Estado:
    return Estado(MAPA_CLASE, INICIO_CLASE)


def actuar(estado: Estado, accion: int, costo_paso: float = -1.0):
    """Devuelve (estado_nuevo, recompensa, terminó)."""
    nuevo = estado.mover(accion)
    recompensa, termino = recompensa_de(nuevo.celda(), costo_paso=costo_paso)
    return nuevo, recompensa, termino


def q_learning(
    estado_inicial: Estado,
    episodios: int = 80,
    pasos: int = 12,
    alpha: float = 0.5,
    gamma: float = 0.95,
    eps: float = 0.2,
    semilla: int = 0,
    costo_paso: float = -1.0,
):
    """Q-learning tabular. El estado es la posición, porque el mapa no cambia.

    Devuelve la tabla Q (dict posición → vector de 4 acciones) y el retorno
    de cada episodio.
    """
    rng = np.random.default_rng(semilla)
    Q: dict[tuple[int, int], np.ndarray] = {}

    def fila(estado: Estado) -> np.ndarray:
        if estado.pos not in Q:
            Q[estado.pos] = np.zeros(4)
        return Q[estado.pos]

    retornos = np.empty(episodios)
    for ep in range(episodios):
        estado = estado_inicial.copia()
        total = 0.0
        for _ in range(pasos):
            if rng.random() < eps:
                accion = int(rng.integers(0, 4))
            else:
                accion = int(np.argmax(fila(estado)))
            siguiente, recompensa, termino = actuar(estado, accion, costo_paso=costo_paso)
            futuro = 0.0 if termino else gamma * float(np.max(fila(siguiente)))
            fila(estado)[accion] += alpha * (recompensa + futuro - fila(estado)[accion])
            total += recompensa
            estado = siguiente
            if termino:
                break
        retornos[ep] = total
    return Q, retornos


def camino_greedy(Q, estado_inicial: Estado, pasos: int = 12, costo_paso: float = -1.0):
    """Sigue la mejor acción de Q. Devuelve (camino, retorno, llegó_al_trofeo)."""
    estado = estado_inicial.copia()
    camino = [estado.pos]
    total = 0.0
    for _ in range(pasos):
        valores = Q.get(estado.pos)
        if valores is None:
            return camino, total, False
        accion = int(np.argmax(valores))
        estado, recompensa, termino = actuar(estado, accion, costo_paso=costo_paso)
        total += recompensa
        camino.append(estado.pos)
        if termino:
            return camino, total, estado.celda() == TROFEO
    return camino, total, False


def media_ultimos(retornos, k: int = 10) -> float:
    """Media de los últimos k retornos. Sirve para leer la curva, no un episodio suelto."""
    serie = np.asarray(retornos, dtype=float)
    if k <= 0:
        raise ValueError("k tiene que ser positivo")
    return float(np.mean(serie[-k:]))
