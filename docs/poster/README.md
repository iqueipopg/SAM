# Póster A0 — SAMA / SAM (CIPIE 2026)

Póster científico A0 del proyecto SAM, importado desde el proyecto de Claude
Design del grupo y preparado para impresión.

## Ficheros

| Fichero | Descripción |
|---|---|
| `SAMA_Poster_A0.html` | Fuente del póster. Autocontenido: las fuentes (Outfit + JetBrains Mono) van embebidas en base64, no depende de red. Se ve a pantalla (se autoescala al viewport) y en impresión sale a A0. Lleva el nombre nuevo ("Systematic Agent for Meta-analysis"). |
| `SAMA_Poster_A0.pdf` | **Original congelado**, tal como se presentó en CIPIE 2026 con el nombre antiguo ("Semi-Automatic Meta-Analysis"). No se regenera. **A0 real (841 × 1189 mm)**, una página, texto vectorial con fuentes embebidas. |
| `SAMA_Poster_A0_updated.pdf` | Versión actualizada con el nombre nuevo ("Systematic Agent for Meta-analysis"), regenerada desde el HTML. Visualmente idéntica al original salvo la expansión del acrónimo. Mismo formato: **A0 real (841 × 1189 mm)**, una página, vectorial. |
| `assets/` | Logos (Comillas horizontal, CIPIE 2026) en alta resolución. |
| `render_pdf.py` | Script que regenera el PDF actualizado (`SAMA_Poster_A0_updated.pdf`) desde el HTML. |

## Nota sobre el nombre del proyecto

El póster **impreso y presentado en CIPIE 2026** usó la expansión original del
acrónimo: **"SAM: Semi-Automatic Meta-Analysis"**. Con posterioridad el proyecto
se renombró a **"SAM: Systematic Agent for Meta-analysis"** (la sigla `SAM` no
cambia; solo cambia lo que significa). Se conservan **ambos PDF**:

- `SAMA_Poster_A0.pdf` — el original, congelado exactamente como se presentó.
- `SAMA_Poster_A0_updated.pdf` — la versión actualizada con el nombre nuevo.

## Impresión

El PDF es **vectorial**, así que se imprime nítido a A0 sin pérdida de calidad.
Enviar `SAMA_Poster_A0.pdf` a la imprenta al 100 % (tamaño A0, sin escalar,
sin márgenes).

## Regenerar el PDF

Tras editar `SAMA_Poster_A0.html`:

```bash
pip install playwright && python -m playwright install chromium   # una vez
python docs/poster/render_pdf.py
```

## Notas técnicas

- El bloque `@media print` escala el layout con **`zoom: 1.88976`** (1682 px → 841 mm),
  no con `transform: scale`, que dispara un bug de recorte en la impresión de Chromium.
- κ (griega), ★, ≈, ² y · caen a una fuente de fallback (Outfit no incluye griego);
  van embebidas y vectoriales, se ven correctas.
