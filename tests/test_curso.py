"""Pruebas rápidas del material de clase. No entrenan el desafío completo."""

import numpy as np

from evolucion import (
    aptitud_mochila,
    evolucionar_bits,
    evolucionar_mochila,
    evolucionar_ruta,
    evolucionar_xor,
    programacion_genetica,
)
from grilla import (
    IZQUIERDA,
    MAPA_CLASE,
    TROFEO,
    VACIO,
    ZOMBI,
    Estado,
    actuar,
    camino_greedy,
    inicio_clase,
    media_ultimos,
    q_learning,
    recompensa_de,
)
from rl_utils import CliffWalking
from salmon_env import SalmonFarmEnv


def test_recompensa_de_la_grilla():
    assert recompensa_de(TROFEO) == (1000.0, True)
    assert recompensa_de(ZOMBI) == (-100.0, True)
    assert recompensa_de(VACIO, costo_paso=-1) == (-1.0, False)


def test_chocar_con_el_borde_cobra_el_paso():
    nuevo, recompensa, termino = actuar(Estado(MAPA_CLASE, (2, 0)), IZQUIERDA)
    assert nuevo.pos == (2, 0)
    assert recompensa == -1.0 and termino is False


def test_qlearning_llega_al_trofeo_por_el_camino_corto():
    estado = inicio_clase()
    Q, retornos = q_learning(estado, semilla=0)
    camino, retorno, llego = camino_greedy(Q, estado)
    assert llego is True
    assert retorno == 997.0
    assert camino == [(2, 0), (2, 1), (1, 1), (0, 1), (0, 0)]
    # ε sigue en 0,2: el entrenamiento todavía pisa algún zombi. La política greedy no.
    assert media_ultimos(retornos) < retorno


def test_algoritmo_genetico_encuentra_los_bits_y_la_mochila():
    objetivo = [1, 0, 1, 1, 0, 0, 1, 0]
    mejor, _ = evolucionar_bits(objetivo, semilla=0)
    assert list(mejor) == objetivo

    pesos, valores, capacidad = [2, 3, 4, 5], [3, 4, 5, 8], 8
    assert aptitud_mochila([0, 1, 0, 1], pesos, valores, capacidad) == 12
    assert aptitud_mochila([1, 1, 1, 1], pesos, valores, capacidad) == 0
    mejor, _ = evolucionar_mochila(pesos, valores, capacidad, semilla=0)
    assert aptitud_mochila(mejor, pesos, valores, capacidad) == 12


def test_la_ruta_evolutiva_llega_al_trofeo():
    _, _, _, retorno, llego = evolucionar_ruta(semilla=0)
    assert llego is True
    assert retorno > 900


def test_programacion_genetica_da_459():
    _, valor, _ = programacion_genetica(semilla=0)
    assert valor == 459


def test_neuroevolucion_resuelve_xor():
    _, exactitud, _, _ = evolucionar_xor(semilla=1)
    assert exactitud == 1.0


def test_cliffwalking_un_paso_cuesta_uno():
    entorno = CliffWalking(seed=0)
    entorno.reset()
    _, recompensa, termino, _ = entorno.step(0)
    assert recompensa == -1.0
    assert termino is False


def test_salmon_observa_cuatro_numeros():
    entorno = SalmonFarmEnv(grupo="G01", seed=0, contar_presupuesto=False)
    obs = entorno.reset()
    assert len(obs) == 4
    obs2, recompensa, termino, _ = entorno.step(1)
    assert len(obs2) == 4
    assert termino is False
    assert np.isfinite(recompensa)
