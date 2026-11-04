# Aprendizaje por Refuerzo: métricas, técnicas y desafío
**Machine Learning · Duoc UC, Escuela de Informática y Telecomunicaciones (Puerto Montt)**

Este repositorio alcanza para la clase. Trae las dos unidades, los módulos de Python y los CSV. En Colab no se sube ningún archivo aparte: la primera celda clona este mismo repositorio.

Material de clase en dos unidades. La **A** deja listo el desafío del centro de cultivo. La **B** es la sesión en que se ve otra forma de buscar una política: algoritmos genéticos, programación genética y neuroevolución, sobre el mismo juego de la grilla. Todo corre con `numpy`, `pandas` y `matplotlib`.

## Unidad A — para el desafío

Van en el orden de la clase.

| # | Notebook | Para qué | Abrir |
|---|---|---|---|
| 01 | La **grilla del héroe** | Q-learning que se ve: un paso de la tabla, y después el trofeo | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/01_qlearning_grilla.ipynb) |
| 02 | Teoría y **métricas** | MDP, Q-learning vs SARSA, semillas, IC95%, AUC, regret, evaluación greedy | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/02_teoria_y_metricas_rl.ipynb) |
| 03 | **Técnicas** resueltas | estado, memoria, Q inicial, Monte Carlo vs TD, offline, robustez | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/03_tecnicas_resueltas.ipynb) |
| 04 | **Ejercicios** con autocomprobación | retorno, bandidos, Bellman, programación dinámica, bugs, significancia | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/04_ejercicios_resueltos.ipynb) |
| 05 | **Desafío** del salmón | trabajo en grupo con el código de grupo | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/05_desafio_salmonicultura.ipynb) |

| Quién | Qué |
|---|---|
| En clase, se muestra | 01 (el camino del héroe), 02 (métricas) y 03 (las seis técnicas, con el resultado en pantalla) |
| Tarea, lo resuelven | los ejercicios en blanco del 01 y el notebook 04. La celda Solución se abre después de intentar |
| Trabajo, lo resuelven | el notebook 05, secciones A–F. El único ejemplo incluido es el agente que cosecha el día 0 y **no** sirve. La solución no está en ese notebook |

## Unidad B — cuando la tabla no alcanza

| # | Notebook | Para qué | Abrir |
|---|---|---|---|
| 06 | **Algoritmo genético** | bits, mochila y una ruta del héroe sin tabla Q | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/06_algoritmo_genetico.ipynb) |
| 07 | **Programación genética** | la política puede ser una expresión, no una tabla | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/07_programacion_genetica.ipynb) |
| 08 | **Neuroevolución** | los pesos de una red se buscan, no se retropropagan. XOR en clase | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/08_neuroevolucion.ipynb) |

Orden: **06 → 07 → 08**. En cada uno se muestra un ejemplo (bits y ruta, la expresión 459, el XOR) y queda un ejercicio en blanco. El 06 vuelve a la grilla del 01: el genético llega al trofeo y a menudo gasta pasos de más; Q-learning se había quedado con el camino corto (retorno 997).

La unidad B adapta la secuencia práctica del diplomado de Alexandre Bergel (Universidad de Chile, 2025). El código de esta carpeta está reescrito para esta clase y no usa Gymnasium ni NEAT, así que corre en el mismo Jupyter del resto del curso.

## Juegos clásicos — para mostrar

No son tarea y no reemplazan el notebook 05. Son el mismo Q-learning de la grilla, en tres juegos que se abren en Colab. La primera celda instala Gymnasium. Los notebooks 01 a 08 no lo necesitan.

| # | Notebook | Para qué | Abrir |
|---|---|---|---|
| 09 | **Lago congelado** | La cuenta de un paso, como el trofeo, y después el hielo que resbala | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/09_lago_congelado.ipynb) |
| 10 | **El taxi** | 500 estados, −1 por paso y +20 al dejar al pasajero | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/10_taxi.ipynb) |
| 11 | **El péndulo** | El estado continuo se corta en cajas. Ahí se ve el límite de la tabla | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/aprendizaje-refuerzo-duoc/blob/main/notebooks/11_cartpole.ipynb) |

Orden para proyectarlos: **09 → 10 → 11**. El 09 repite la cuenta a mano del 01 (la meta deja 0,800 y el paso anterior, 0,608). El 10 es el mismo ciclo con un tablero en texto. El 11 corta el ángulo en tramos, que es la agregación de estado del notebook 03, y cierra con el motivo del notebook 08.

## Cómo usarlo
**En Colab:** clic en el botón, luego *Entorno de ejecución → Ejecutar todas*. En los notebooks 01 a 08, la primera celda clona este repositorio. Con eso llegan `grilla.py`, `rl_utils.py`, `salmon_env.py`, `evolucion.py` y la carpeta `data/`, y después se instala `requirements.txt` (numpy, pandas, matplotlib y Gymnasium). No hay que subir un zip ni ningún archivo suelto. Los notebooks 09, 10 y 11 instalan Gymnasium, numpy y matplotlib por su cuenta, así que el juego abre aunque no se haya clonado el resto. Para guardar cambios, *Archivo → Guardar una copia en Drive*.

Las celdas que parten con *Solución* quedan plegadas: hay que abrirlas a propósito. Lo que se proyecta en clase ya viene con el resultado.

En Colab los archivos generados (bitácora, entrega) se borran al cerrar la sesión. El notebook 05 tiene una celda que los empaqueta y descarga.

Los ejercicios traen una celda **Comprobar**. Si la función sigue en blanco, esa celda avisa y el resto del notebook sigue. La celda **Solución** está detrás de un título: doble clic para abrirla. Conviene intentar antes.

**En el computador, con [uv](https://docs.astral.sh/uv/):**
```bash
git clone https://github.com/Giocrisrai/aprendizaje-refuerzo-duoc.git
cd aprendizaje-refuerzo-duoc
uv sync --group dev
uv run pytest
uv run jupyter notebook notebooks/
```

`uv sync` crea `.venv` e instala numpy, pandas, matplotlib y Gymnasium. `--group dev` agrega pytest y Jupyter, que en Colab no hacen falta. `pytest` revisa la grilla, la mochila, la expresión que da 459, el XOR, un paso de Cliff Walking y una observación del centro de salmón. No entrena el desafío ni los juegos.

Sin uv, desde la carpeta `notebooks/`: `pip install -r ../requirements.txt`.

## Dónde está cada idea

Los notebooks 01, 02 y 03 quedan resueltos: se proyectan. La tarea es el 04 y los ejercicios en blanco del 01, 06, 07 y 08. El trabajo es el 05.

| Si se traban en | Vuelvan a |
|---|---|
| Estado, acción, recompensa, la tabla Q, α, γ y ε | Notebook 01. Ahí la cuenta sale con números: 500, 237 y 750 |
| Entrenamiento distinto de la política greedy, semillas, IC95% | Notebook 02 |
| Estado grande, memoria, Q en cero, Monte Carlo, datos de otro centro | Notebook 03 |
| Practicar la cuenta antes de medir | Notebook 04 |
| El centro de salmón | Notebook 05. Cada sección apunta a la clase donde eso ya se vio |
| Otra forma de buscar la misma ruta del héroe | Notebooks 06, 07 y 08 |
| Ver el mismo ciclo en un juego | Notebooks 09, 10 y 11. No reemplazan el 05 |

| Palabra | En esta clase quiere decir |
|---|---|
| Retorno | La suma de las recompensas de un episodio. En la grilla, el camino corto suma 997 |
| Q(casilla, acción) | Lo que se espera sumar si se toma esa acción y después se juega bien. No es la recompensa del paso |
| α | Cuánto se acerca la celda al objetivo nuevo. En la grilla vale 0,5: se queda a mitad de camino |
| γ | Cuánto vale el futuro. En la grilla vale 0,95: cada paso de distancia le quita un 5% |
| ε | Probabilidad de actuar al azar mientras se entrena. La política que se entrega va con ε = 0 |
| Greedy | Sin azar: la acción de mayor Q |

## Archivos
```
notebooks/01_qlearning_grilla.ipynb     la clase que se proyecta primero
notebooks/02_teoria_y_metricas_rl.ipynb Cliff Walking, Q-learning y SARSA, cómo se reporta
notebooks/03_tecnicas_resueltas.ipynb   las seis técnicas, con el resultado en pantalla
notebooks/04_ejercicios_resueltos.ipynb tarea, con celda Comprobar
notebooks/05_desafio_salmonicultura.ipynb trabajo en grupo
notebooks/06_algoritmo_genetico.ipynb
notebooks/07_programacion_genetica.ipynb
notebooks/08_neuroevolucion.ipynb
notebooks/09_lago_congelado.ipynb      FrozenLake. La misma cuenta que el trofeo
notebooks/10_taxi.ipynb                Taxi. 500 estados, se ve el tablero
notebooks/11_cartpole.ipynb            CartPole. El estado se corta en cajas
notebooks/grilla.py        el mapa del héroe. Lo usan el 01 y el 06
notebooks/rl_utils.py      Cliff Walking, los agentes y las métricas. Lo usan el 02 y el 04
notebooks/salmon_env.py    el simulador del centro. Lo usa el 05
notebooks/evolucion.py     genético, programación genética y neuroevolución. Lo usan el 06, el 07 y el 08
data/                      corridas ya hechas de Cliff Walking, inventario y operador
tests/                     pytest. No entrena el desafío
```

## Fuentes y videos

**Libro base (gratis)**
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2ª ed. [PDF](http://incompleteideas.net/book/the-book-2nd.html). Cliff Walking es el ejemplo 6.6. Q-learning y SARSA están en el capítulo 6.

**Para llegar a clase**
- [StatQuest — Reinforcement Learning: Essential Concepts](https://www.youtube.com/watch?v=Z-T0iJEXiwM) (18 min, inglés con subtítulos).
- [Q-Learning Explained, Serrano.Academy](https://www.youtube.com/watch?v=qhRNvCVVJaA).
- En español: [Codificando Bits, nivel básico](https://codificandobits.com/curso/aprendizaje-por-refuerzo-nivel-basico/) y, para SARSA, Q-learning y Monte Carlo, el [nivel intermedio](https://codificandobits.com/curso/aprendizaje-por-refuerzo-nivel-intermedio/).

**Curso universitario**
- David Silver: [clase 2, MDP](https://www.youtube.com/watch?v=lfHX2hHRMVQ) · [clase 4, Monte Carlo](https://www.youtube.com/watch?v=PnHCvfgC_ZA) · [clase 5, Q-learning y SARSA](https://www.youtube.com/watch?v=0g4j2k_Ggc4).

**El mismo ciclo, en el curso de Hugging Face** (inglés). El notebook 01 lo recorre en español con la grilla, sin Gymnasium. El notebook 09 es ese lago, ya resuelto y con la cuenta de un paso.
- [Unidad 1, el ciclo estado–acción–recompensa](https://huggingface.co/learn/deep-rl-course/unit1/rl-framework).
- [Unidad 2, Q-learning y la tabla](https://huggingface.co/learn/deep-rl-course/unit2/q-learning).
- [Práctica en FrozenLake](https://huggingface.co/learn/deep-rl-course/unit2/hands-on): ahí programan el bucle. Aquí el bucle ya está escrito: la cuenta de un paso está en el notebook 01 y, con el lago, en el 09.

**Para citar en un informe**
- Henderson et al. (2018), [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560).
- Agarwal et al. (2021), [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264).
- Levine et al. (2020), [Offline RL: Tutorial, Review, and Perspectives](https://arxiv.org/abs/2005.01643).
