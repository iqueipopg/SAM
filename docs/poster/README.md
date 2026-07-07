# Póster A0 — SAMA / SAM (CIPIE 2026)

Póster científico A0 del proyecto SAM, importado desde el proyecto de Claude
Design del grupo y preparado para impresión.

## Ficheros

| Fichero | Descripción |
|---|---|
| `SAMA_Poster_A0.html` | Fuente del póster. Autocontenido: las fuentes (Outfit + JetBrains Mono) van embebidas en base64, no depende de red. Se ve a pantalla (se autoescala al viewport) y en impresión sale a A0. |
| `SAMA_Poster_A0.pdf` | PDF final listo para imprimir. **A0 real (841 × 1189 mm)**, una página, texto vectorial con fuentes embebidas. |
| `assets/` | Logos (Comillas horizontal, CIPIE 2026) en alta resolución. |
| `render_pdf.py` | Script que regenera el PDF desde el HTML. |

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
