import { useMemo, useState } from 'react'
import { ArrowUp, ArrowDown, Search as SearchIcon } from 'lucide-react'

// ── Helpers ──────────────────────────────────────────────────────────────────

const getAuthorYear = (paper) => {
  const authors = paper.authors || paper.author || ''
  const year = paper.year || paper.publication_year || ''
  const firstAuthor =
    (Array.isArray(authors) ? authors[0] : String(authors).split(',')[0] || '')
      .trim() || 'Unknown'
  const lastName = firstAuthor.split(' ').pop() || firstAuthor
  return year ? `${lastName} ${year}` : lastName
}

const getDesign = (paper) => {
  const sc = paper.study_characteristics || {}
  return sc.study_design || paper.design || paper.study_design || '—'
}

const getN = (paper) => {
  const sc = paper.study_characteristics || {}
  const n =
    sc.sample_size ??
    sc.total_n ??
    sc.n_total ??
    paper.sample_size ??
    paper.n_total ??
    null
  if (n === null || n === undefined || n === '') return null
  const num = Number(n)
  return Number.isFinite(num) ? num : null
}

const getIntervention = (paper) => {
  const iv = paper.intervention || {}
  return (
    iv.intervention_name ||
    iv.name ||
    iv.description ||
    paper.intervention_name ||
    '—'
  )
}

const getOutcomes = (paper) => {
  const outs = paper.outcomes || paper.outcome_measures || []
  if (!Array.isArray(outs)) return []
  return outs
    .map((o) => (typeof o === 'string' ? o : o.name || o.outcome || o.measure))
    .filter(Boolean)
}

const getQualityWeight = (paper) => {
  const w = paper.quality_weight ?? paper.quality_assessment?.weight
  if (w === null || w === undefined) return null
  const num = Number(w)
  return Number.isFinite(num) ? num : null
}

const getQualityBadge = (paper) => {
  const qa = paper.quality_assessment || {}
  return qa.overall || null
}

const weightTone = (w) => {
  if (w === null) return 'bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400'
  if (w >= 0.8) return 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300'
  if (w >= 0.5) return 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300'
  return 'bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300'
}

// ── Component ────────────────────────────────────────────────────────────────

export default function StudiesTable({ papers = [] }) {
  const [sortKey, setSortKey] = useState('study')
  const [sortDir, setSortDir] = useState('asc')
  const [filter, setFilter] = useState('')

  const rows = useMemo(() => {
    return (papers || []).map((p, idx) => ({
      idx,
      raw: p,
      study: getAuthorYear(p),
      design: getDesign(p),
      n: getN(p),
      intervention: getIntervention(p),
      outcomes: getOutcomes(p),
      weight: getQualityWeight(p),
      badge: getQualityBadge(p),
    }))
  }, [papers])

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase()
    if (!q) return rows
    return rows.filter((r) => {
      const hay = [
        r.study,
        r.design,
        r.intervention,
        r.outcomes.join(' '),
        r.badge || '',
        r.raw.title || '',
        r.raw.pmid || '',
      ]
        .join(' ')
        .toLowerCase()
      return hay.includes(q)
    })
  }, [rows, filter])

  const sorted = useMemo(() => {
    const copy = [...filtered]
    copy.sort((a, b) => {
      let av = a[sortKey]
      let bv = b[sortKey]
      if (sortKey === 'outcomes') {
        av = a.outcomes.length
        bv = b.outcomes.length
      }
      // Null handling: nulls at the end (regardless of direction)
      const aNull = av === null || av === undefined || av === ''
      const bNull = bv === null || bv === undefined || bv === ''
      if (aNull && bNull) return 0
      if (aNull) return 1
      if (bNull) return -1
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortDir === 'asc' ? av - bv : bv - av
      }
      const as = String(av).toLowerCase()
      const bs = String(bv).toLowerCase()
      if (as < bs) return sortDir === 'asc' ? -1 : 1
      if (as > bs) return sortDir === 'asc' ? 1 : -1
      return 0
    })
    return copy
  }, [filtered, sortKey, sortDir])

  const toggleSort = (key) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const columns = [
    { key: 'study', label: 'Study' },
    { key: 'design', label: 'Design' },
    { key: 'n', label: 'N', align: 'right' },
    { key: 'intervention', label: 'Intervention' },
    { key: 'outcomes', label: 'Outcomes' },
    { key: 'weight', label: 'Quality', align: 'right' },
  ]

  if (!papers || papers.length === 0) {
    return (
      <p className="text-sm text-zinc-500 dark:text-zinc-400 italic">
        No hay estudios extraídos disponibles.
      </p>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="relative flex-1 min-w-[200px] max-w-md">
          <SearchIcon
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400 dark:text-zinc-500"
          />
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filtrar por autor, diseño, intervención, outcome..."
            className="w-full pl-9 pr-3 py-2 rounded-lg border border-zinc-200 bg-white text-sm text-zinc-700 placeholder:text-zinc-400 focus:outline-none focus:border-sam-400 focus:ring-2 focus:ring-sam-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:placeholder:text-zinc-500 dark:focus:ring-sam-900/40"
          />
        </div>
        <span className="text-xs text-zinc-400 dark:text-zinc-500 font-mono">
          {sorted.length} / {rows.length}
        </span>
      </div>

      <div className="overflow-x-auto rounded-xl border border-zinc-200 dark:border-zinc-700">
        <table className="min-w-full text-sm">
          <thead className="bg-zinc-50 dark:bg-zinc-800/60 text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
            <tr>
              {columns.map((c) => (
                <th
                  key={c.key}
                  onClick={() => toggleSort(c.key)}
                  className={`px-3 py-2.5 font-semibold cursor-pointer select-none hover:text-zinc-700 dark:hover:text-zinc-200 transition-colors ${
                    c.align === 'right' ? 'text-right' : 'text-left'
                  }`}
                >
                  <span className="inline-flex items-center gap-1">
                    {c.label}
                    {sortKey === c.key &&
                      (sortDir === 'asc' ? (
                        <ArrowUp size={11} />
                      ) : (
                        <ArrowDown size={11} />
                      ))}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
            {sorted.map((r) => (
              <tr
                key={r.idx}
                className="bg-white hover:bg-zinc-50 dark:bg-zinc-900/40 dark:hover:bg-zinc-800/60 transition-colors"
              >
                <td className="px-3 py-2.5 align-top">
                  <div className="font-medium text-zinc-900 dark:text-zinc-100">
                    {r.study}
                  </div>
                  {r.raw.pmid && (
                    <div className="text-[11px] text-zinc-400 dark:text-zinc-500 font-mono">
                      PMID {r.raw.pmid}
                    </div>
                  )}
                </td>
                <td className="px-3 py-2.5 align-top text-zinc-600 dark:text-zinc-300">
                  {r.design}
                </td>
                <td className="px-3 py-2.5 align-top text-right font-mono text-zinc-700 dark:text-zinc-200">
                  {r.n ?? '—'}
                </td>
                <td className="px-3 py-2.5 align-top text-zinc-600 dark:text-zinc-300 max-w-xs">
                  <span className="line-clamp-2">{r.intervention}</span>
                </td>
                <td className="px-3 py-2.5 align-top">
                  {r.outcomes.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {r.outcomes.slice(0, 3).map((o, i) => (
                        <span
                          key={i}
                          className="inline-block px-1.5 py-0.5 rounded text-[11px] bg-sam-50 text-sam-700 border border-sam-100 dark:bg-sam-900/30 dark:text-sam-300 dark:border-sam-800/60"
                        >
                          {String(o).slice(0, 40)}
                        </span>
                      ))}
                      {r.outcomes.length > 3 && (
                        <span className="text-[11px] text-zinc-400 dark:text-zinc-500">
                          +{r.outcomes.length - 3}
                        </span>
                      )}
                    </div>
                  ) : (
                    <span className="text-zinc-400 dark:text-zinc-500">—</span>
                  )}
                </td>
                <td className="px-3 py-2.5 align-top text-right">
                  <span
                    className={`inline-block px-2 py-0.5 rounded font-mono text-[11px] font-semibold ${weightTone(
                      r.weight
                    )}`}
                    title={r.badge || ''}
                  >
                    {r.weight !== null ? r.weight.toFixed(2) : '—'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
