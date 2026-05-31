# Composición del subconjunto T17.C.2b (N=300)

Subconjunto determinista y congelado, fuente única compartida por los
dos tracks de T17.C.2b. Generado sin LLM por `scripts/build_c2b_subset.py`.

## Parámetros

- Seed: `20260531` (`random.Random(20260531)`).
- Celdas (diseño balanceado): 150 relevant / 150 irrelevant.
- Muestreo: pools ordenados por `covidence_id`, luego `rng.sample(pool, 150)` por clase; unión ordenada por `covidence_id`.

## Totales

- Total filas: 300.
- Con abstract (criterio gold): 300.
- Title-only (sin abstract): 0.
- `human_label_ta`: relevant = 150, irrelevant = 150.

## Pools candidatos (sample_medio_a3 + abstract gold)

- relevant con-abstract: 256
- irrelevant con-abstract: 991
- relevant sin-abstract: 744
- irrelevant sin-abstract: 9

## Criterio "con-abstract"

Réplica exacta de `eval_hero1_screening._build_papers` y
`eval_a3_claude._load_candidates`: el abstract se toma del GOLD
(`gold_screening_unified.csv`), se aplica `.strip()`, y se considera
presente solo si no queda vacío y su `lower()` no está en ("nan", "none").
Es el mismo criterio que usará la evaluación cuantitativa.

## Decisión: exclusión de title-only

Se excluyen del subconjunto cuantitativo los papers sin abstract
(title-only). Rationale: los pools sin-abstract están radicalmente
desbalanceados — 744 relevant sin-abstract frente a solo 9 irrelevant
sin-abstract —, lo que hace imposible equilibrar 150/150 en la celda
title-only sin recortar muy por debajo de N. La política de tratamiento
de title-only sí está igualada en el prompt `c18ab41` (ambos tracks lo
manejan idénticamente), pero queda FUERA de la evaluación cuantitativa
de C.2b para preservar el balance de clases y la paridad de entrada.
