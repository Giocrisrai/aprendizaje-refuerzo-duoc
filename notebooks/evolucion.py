"""Búsqueda evolutiva para la segunda unidad de la clase.

Se usa en los notebooks 06 (bits, mochila y ruta), 07 (expresiones) y 08 (XOR).
Los tres comparten el mismo esqueleto: población, aptitud, selección y variación.

- un algoritmo genético sobre bits, una mochila y una ruta del héroe;
- programación genética de expresiones aritméticas;
- neuroevolución de una red chica para el XOR.

No usa Gymnasium ni NEAT: corre en el mismo entorno que el resto del curso
(`numpy`) y termina en segundos. CartPole y el caminante del diplomado son
el mismo algoritmo con otro ambiente; no hacen falta para la clase.
"""

from __future__ import annotations

import numpy as np

from grilla import Estado, MAPA_CLASE, INICIO_CLASE, TROFEO, actuar


def _torneo(rng, aptitud: np.ndarray, k: int = 3) -> int:
    candidatos = rng.integers(0, len(aptitud), size=k)
    return int(candidatos[np.argmax(aptitud[candidatos])])


def evolucionar_bits(
    objetivo,
    poblacion: int = 80,
    generaciones: int = 40,
    p_mut: float = 0.05,
    semilla: int = 0,
):
    """Busca una cadena de bits. La aptitud es cuántos bits coinciden."""
    rng = np.random.default_rng(semilla)
    meta = np.asarray(objetivo, dtype=int)
    n = len(meta)
    pop = rng.integers(0, 2, size=(poblacion, n))
    historia = []
    mejor = pop[0].copy()
    mejor_fit = -1
    for _ in range(generaciones):
        aptitud = (pop == meta).sum(axis=1).astype(float)
        historia.append(int(aptitud.max()))
        i = int(np.argmax(aptitud))
        if aptitud[i] > mejor_fit:
            mejor_fit = int(aptitud[i])
            mejor = pop[i].copy()
        if mejor_fit == n:
            break
        nueva = np.empty_like(pop)
        for j in range(poblacion):
            a = pop[_torneo(rng, aptitud)]
            b = pop[_torneo(rng, aptitud)]
            corte = int(rng.integers(1, n)) if n > 1 else 1
            hijo = np.concatenate([a[:corte], b[corte:]])
            muta = rng.random(n) < p_mut
            hijo[muta] = 1 - hijo[muta]
            nueva[j] = hijo
        pop = nueva
    return mejor, np.asarray(historia, dtype=int)


def aptitud_mochila(bits, pesos, valores, capacidad: float) -> float:
    """Suma de valores si el peso cabe. Si se pasa, aptitud 0.

    `bits` es 0/1, un gen por objeto. Esta función es la que completan
    en el notebook 06; aquí está la versión que usa el algoritmo y las pruebas.
    """
    mascara = np.asarray(bits, dtype=float)
    pesos = np.asarray(pesos, dtype=float)
    valores = np.asarray(valores, dtype=float)
    if mascara.shape != pesos.shape or pesos.shape != valores.shape:
        raise ValueError("bits, pesos y valores tienen que medir lo mismo")
    if float(mascara @ pesos) > capacidad:
        return 0.0
    return float(mascara @ valores)


def evolucionar_mochila(
    pesos,
    valores,
    capacidad: float,
    poblacion: int = 40,
    generaciones: int = 30,
    p_mut: float = 0.08,
    semilla: int = 0,
):
    rng = np.random.default_rng(semilla)
    n = len(pesos)
    pop = rng.integers(0, 2, size=(poblacion, n))
    historia = []
    mejor = pop[0].copy()
    mejor_fit = -1.0
    for _ in range(generaciones):
        aptitud = np.array([aptitud_mochila(ind, pesos, valores, capacidad) for ind in pop])
        historia.append(float(aptitud.max()))
        i = int(np.argmax(aptitud))
        if aptitud[i] > mejor_fit:
            mejor_fit = float(aptitud[i])
            mejor = pop[i].copy()
        nueva = np.empty_like(pop)
        for j in range(poblacion):
            a = pop[_torneo(rng, aptitud)]
            b = pop[_torneo(rng, aptitud)]
            corte = int(rng.integers(1, n)) if n > 1 else 1
            hijo = np.concatenate([a[:corte], b[corte:]]).astype(int)
            muta = rng.random(n) < p_mut
            hijo[muta] = 1 - hijo[muta]
            nueva[j] = hijo
        pop = nueva
    return mejor, np.asarray(historia)


def _jugar_ruta(acciones, costo_paso: float = -1.0):
    estado = Estado(MAPA_CLASE, INICIO_CLASE)
    total = 0.0
    camino = [estado.pos]
    for accion in acciones:
        estado, recompensa, termino = actuar(estado, int(accion), costo_paso=costo_paso)
        total += recompensa
        camino.append(estado.pos)
        if termino:
            return camino, total, estado.celda() == TROFEO
    return camino, total, False


def evolucionar_ruta(
    n_pasos: int = 6,
    poblacion: int = 60,
    generaciones: int = 25,
    p_mut: float = 0.15,
    semilla: int = 0,
):
    """El cromosoma es la secuencia de acciones. No hay tabla Q."""
    rng = np.random.default_rng(semilla)
    pop = rng.integers(0, 4, size=(poblacion, n_pasos))
    historia = []
    mejor = pop[0].copy()
    mejor_fit = -1e18
    for _ in range(generaciones):
        resultados = [_jugar_ruta(ind) for ind in pop]
        aptitud = np.array([r[1] for r in resultados])
        historia.append(float(aptitud.max()))
        i = int(np.argmax(aptitud))
        if aptitud[i] > mejor_fit:
            mejor_fit = float(aptitud[i])
            mejor = pop[i].copy()
        nueva = np.empty_like(pop)
        for j in range(poblacion):
            a = pop[_torneo(rng, aptitud)]
            b = pop[_torneo(rng, aptitud)]
            corte = int(rng.integers(1, n_pasos))
            hijo = np.concatenate([a[:corte], b[corte:]])
            muta = rng.random(n_pasos) < p_mut
            hijo[muta] = rng.integers(0, 4, size=int(muta.sum()))
            nueva[j] = hijo
        pop = nueva
    camino, retorno, llego = _jugar_ruta(mejor)
    return mejor, np.asarray(historia), camino, retorno, llego


# ---------------------------------------------------------------------------
# Programación genética: expresiones con + − ×
# ---------------------------------------------------------------------------

def arbol_aleatorio(rng, profundidad: int, terminales):
    if profundidad <= 0 or rng.random() < 0.35:
        return ("n", int(rng.choice(terminales)))
    op = str(rng.choice(("+", "-", "*")))
    return (
        "op",
        op,
        arbol_aleatorio(rng, profundidad - 1, terminales),
        arbol_aleatorio(rng, profundidad - 1, terminales),
    )


def evaluar(arbol) -> float:
    if arbol[0] == "n":
        return float(arbol[1])
    _, op, izq, der = arbol
    a, b = evaluar(izq), evaluar(der)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError(op)


def texto_arbol(arbol) -> str:
    if arbol[0] == "n":
        return str(arbol[1])
    return f"({texto_arbol(arbol[2])} {arbol[1]} {texto_arbol(arbol[3])})"


def n_nodos(arbol) -> int:
    if arbol[0] == "n":
        return 1
    return 1 + n_nodos(arbol[2]) + n_nodos(arbol[3])


def aptitud_expresion(arbol, objetivo: float) -> float:
    """Más alto es mejor. El tamaño desempata para que no crezca un árbol enorme."""
    return -abs(evaluar(arbol) - objetivo) - 0.01 * n_nodos(arbol)


def _reemplazar(arbol, indice: int, nuevo):
    contador = {"i": 0}

    def rec(nodo):
        i = contador["i"]
        contador["i"] += 1
        if i == indice:
            return nuevo
        if nodo[0] == "n":
            return nodo
        return ("op", nodo[1], rec(nodo[2]), rec(nodo[3]))

    return rec(arbol)


def programacion_genetica(
    objetivo: float = 459,
    poblacion: int = 200,
    generaciones: int = 25,
    profundidad: int = 4,
    semilla: int = 0,
    terminales=range(1, 10),
):
    rng = np.random.default_rng(semilla)
    terminales = list(terminales)
    pop = [arbol_aleatorio(rng, profundidad, terminales) for _ in range(poblacion)]
    historia = []
    mejor = pop[0]
    mejor_fit = -1e18
    for _ in range(generaciones):
        aptitud = np.array([aptitud_expresion(ind, objetivo) for ind in pop])
        i = int(np.argmax(aptitud))
        historia.append(abs(evaluar(pop[i]) - objetivo))
        if aptitud[i] > mejor_fit:
            mejor_fit = float(aptitud[i])
            mejor = pop[i]
        if abs(evaluar(mejor) - objetivo) < 1e-9:
            break
        nueva = []
        for _j in range(poblacion):
            a = pop[_torneo(rng, aptitud)]
            b = pop[_torneo(rng, aptitud)]
            hijo = _reemplazar(a, int(rng.integers(0, n_nodos(a))), _subarbol_al_azar(rng, b))
            if rng.random() < 0.3:
                hijo = _reemplazar(
                    hijo,
                    int(rng.integers(0, n_nodos(hijo))),
                    arbol_aleatorio(rng, 2, terminales),
                )
            nueva.append(hijo)
        pop = nueva
    return mejor, float(evaluar(mejor)), np.asarray(historia)


def _subarbol_al_azar(rng, arbol):
    indice = int(rng.integers(0, n_nodos(arbol)))
    encontrado = {"nodo": None, "i": 0}

    def rec(nodo):
        i = encontrado["i"]
        encontrado["i"] += 1
        if i == indice:
            encontrado["nodo"] = nodo
        if nodo[0] == "op" and encontrado["nodo"] is None:
            rec(nodo[2])
            if encontrado["nodo"] is None:
                rec(nodo[3])

    rec(arbol)
    return encontrado["nodo"]


# ---------------------------------------------------------------------------
# Neuroevolución: pesos de una red, sin retropropagación
# ---------------------------------------------------------------------------

def _sigmoide(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _partir_pesos_xor(pesos, n_ocultas: int = 6):
    pesos = np.asarray(pesos, dtype=float)
    n_w1 = 2 * n_ocultas
    n_b1 = n_ocultas
    n_w2 = n_ocultas
    w1 = pesos[:n_w1].reshape(2, n_ocultas)
    b1 = pesos[n_w1:n_w1 + n_b1]
    w2 = pesos[n_w1 + n_b1:n_w1 + n_b1 + n_w2].reshape(n_ocultas, 1)
    b2 = pesos[-1]
    return w1, b1, w2, b2


def predecir_xor(pesos, entradas, n_ocultas: int = 6):
    w1, b1, w2, b2 = _partir_pesos_xor(pesos, n_ocultas)
    oculta = _sigmoide(np.asarray(entradas, dtype=float) @ w1 + b1)
    return _sigmoide(oculta @ w2 + b2).ravel()


def evolucionar_xor(
    generaciones: int = 60,
    poblacion: int = 80,
    n_ocultas: int = 6,
    semilla: int = 1,
):
    """Maximiza -MSE en las cuatro filas de XOR. Devuelve pesos, exactitud e historia."""
    rng = np.random.default_rng(semilla)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    n_pesos = 2 * n_ocultas + n_ocultas + n_ocultas + 1
    pop = rng.normal(0, 1, size=(poblacion, n_pesos))
    sigma = 0.45
    historia = []
    mejor = pop[0].copy()
    mejor_fit = -1e18

    def aptitud_de(individuo):
        pred = predecir_xor(individuo, X, n_ocultas)
        return -float(np.mean((pred - y) ** 2))

    for _ in range(generaciones):
        aptitud = np.array([aptitud_de(ind) for ind in pop])
        historia.append(float(-aptitud.max()))
        i = int(np.argmax(aptitud))
        if aptitud[i] > mejor_fit:
            mejor_fit = float(aptitud[i])
            mejor = pop[i].copy()
        pred = predecir_xor(mejor, X, n_ocultas)
        if np.mean((pred > 0.5).astype(float) == y) == 1.0 and mejor_fit > -0.02:
            break
        nueva = np.empty_like(pop)
        for j in range(poblacion):
            a = pop[_torneo(rng, aptitud)]
            b = pop[_torneo(rng, aptitud)]
            mascara = rng.random(n_pesos) < 0.5
            hijo = np.where(mascara, a, b)
            hijo = hijo + rng.normal(0, sigma, size=n_pesos)
            nueva[j] = hijo
        # elitismo: el mejor no se pierde
        nueva[0] = mejor
        pop = nueva
        sigma = max(0.05, sigma * 0.97)
    pred = predecir_xor(mejor, X, n_ocultas)
    exactitud = float(np.mean((pred > 0.5).astype(float) == y))
    return mejor, exactitud, pred, np.asarray(historia)


def exactitud_xor(pesos, n_ocultas: int = 6) -> float:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    pred = predecir_xor(pesos, X, n_ocultas)
    return float(np.mean((pred > 0.5).astype(float) == y))
