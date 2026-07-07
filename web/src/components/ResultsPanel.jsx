// Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
// SPDX-License-Identifier: Apache-2.0
import { useState, useEffect } from 'react'
import {
  ScrollText, BarChart3, ShieldCheck, FileOutput, FileText, Table2,
  Sparkles, Download, ChevronDown, ChevronRight, Loader2,
} from 'lucide-react'
import StudiesTable from './StudiesTable'

export default function ResultsPanel({ fullData, runId, apiUrl = '' }) {
  if (!fullData) return null

  const {
    protocol,
    prisma,
    quality_assessment: a6,
    synthesis: a7,
    manuscript: a8,
    explainability: a9,
    data_extraction: a5,
  } = fullData

  // A5 papers live under data_extraction.papers or extractions — be lenient
  const extractedPapers =
    a5?.papers || a5?.extractions || fullData.extractions || []

  const pathToUrl = (absPath) => {
    if (!absPath || !runId) return null
    const marker = `outputs/${runId}/`
    const idx = absPath.indexOf(marker)
    if (idx === -1) {
      if (absPath.startsWith('plots/') || !absPath.includes('/')) {
        return `${apiUrl}/api/download/${runId}/${absPath}`
      }
      return null
    }
    return `${apiUrl}/api/download/${runId}/${absPath.substring(idx + marker.length)}`
  }

  return (
    <div className="px-6 py-6 border-t border-zinc-100 dark:border-zinc-800 bg-white dark:bg-zinc-900 space-y-4">
      <p className="text-xs font-semibold tracking-[0.1em] uppercase text-zinc-400 dark:text-zinc-500">
        Results
      </p>
      {a9 && <A9Section a9={a9} runId={runId} apiUrl={apiUrl} />}
      {a7 && <A7Section a7={a7} pathToUrl={pathToUrl} />}
      {extractedPapers.length > 0 && (
        <StudiesSection papers={extractedPapers} />
      )}
      {prisma && <PRISMASection prisma={prisma} pathToUrl={pathToUrl} />}
      {a6 && <A6Section a6={a6} pathToUrl={pathToUrl} />}
      {a8 && <A8Section a8={a8} pathToUrl={pathToUrl} />}
      {protocol && <ProtocolSection protocol={protocol} />}
    </div>
  )
}

// ── A9 Explainability (featured) ─────────────────────────────────────────────

function A9Section({ a9, runId, apiUrl }) {
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const stats = a9?.stats || {}

  const loadPreview = async () => {
    if (preview !== null) return
    setLoading(true)
    try {
      const res = await fetch(`${apiUrl}/api/download/${runId}/explainability.md`)
      const text = await res.text()
      setPreview(text)
    } catch {
      setPreview('(no se pudo cargar el informe)')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-2xl border-2 border-teal-200 bg-gradient-to-br from-teal-50 to-white overflow-hidden">
      <div className="p-5 flex items-start gap-4">
        <div className="w-10 h-10 rounded-xl bg-teal-600 flex items-center justify-center shrink-0">
          <ScrollText size={18} className="text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-teal-900 tracking-tight">
              A9 · Explainability & Traceability
            </h3>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-teal-100 text-teal-700 text-[10px] font-semibold">
              <Sparkles size={10} /> NEW
            </span>
          </div>
          <p className="mt-1 text-sm text-zinc-600">
            Informe en español con la trazabilidad de cada decisión del pipeline.
            Excel con los datos brutos por paper para auditoría estadística.
          </p>

          <div className="mt-4 grid grid-cols-3 gap-3">
            <StatCard label="Papers totales" value={stats.total_papers} />
            <StatCard label="Incluidos" value={stats.included} tone="emerald" />
            <StatCard label="Excluidos" value={stats.excluded} tone="rose" />
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <a
              href={`${apiUrl}/api/download/${runId}/papers_raw.xlsx`}
              download
              className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-teal-600 text-white text-sm font-medium hover:bg-teal-700 transition-colors"
            >
              <Download size={14} /> Papers Excel
            </a>
            <a
              href={`${apiUrl}/api/download/${runId}/explainability.md`}
              download
              className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-white border border-teal-300 text-teal-700 text-sm font-medium hover:bg-teal-50 transition-colors"
            >
              <Download size={14} /> Informe (.md)
            </a>
            <button
              onClick={loadPreview}
              className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-white border border-zinc-200 text-zinc-600 text-sm font-medium hover:border-teal-300 hover:text-teal-700 transition-colors"
            >
              {loading ? <Loader2 size={14} className="animate-spin" /> : <ChevronDown size={14} />}
              {preview !== null ? 'Preview cargado' : 'Ver preview'}
            </button>
          </div>

          {preview !== null && (
            <pre className="mt-4 p-4 rounded-lg bg-zinc-900 text-zinc-100 text-xs leading-relaxed overflow-auto max-h-96 whitespace-pre-wrap font-mono">
              {preview}
            </pre>
          )}
          {stats.elapsed_s !== undefined && (
            <p className="mt-3 text-[11px] text-zinc-400 font-mono">
              Generado en {stats.elapsed_s}s
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, tone = 'teal' }) {
  const tones = {
    teal: 'text-teal-700 bg-teal-50 border-teal-100',
    emerald: 'text-emerald-700 bg-emerald-50 border-emerald-100',
    rose: 'text-rose-700 bg-rose-50 border-rose-100',
  }
  return (
    <div className={`p-3 rounded-lg border ${tones[tone]}`}>
      <div className="font-mono text-xl font-bold">{value ?? 0}</div>
      <div className="text-[11px] opacity-70 mt-0.5">{label}</div>
    </div>
  )
}

// ── A7 Synthesis ─────────────────────────────────────────────────────────────

function A7Section({ a7, pathToUrl }) {
  const analyses = a7?.analyses || {}
  const stats = a7?.stats || {}
  const hasAnalyses = Object.keys(analyses).length > 0

  return (
    <Accordion
      icon={<BarChart3 size={16} className="text-sky-600" />}
      title="A7 · Meta-Analysis Synthesis"
      subtitle={
        hasAnalyses
          ? `${stats.papers_with_usable_data || 0} papers · ${stats.analyses_performed || 0} analyses`
          : 'No quantitative synthesis'
      }
      defaultOpen={true}
    >
      {!hasAnalyses && (
        <p className="text-sm text-zinc-500 italic">
          No meta-analyses performed (need ≥2 papers with usable numeric data).
        </p>
      )}
      {Object.entries(analyses).map(([measure, analysis]) => (
        <MeasureBlock key={measure} measure={measure} analysis={analysis} pathToUrl={pathToUrl} />
      ))}
    </Accordion>
  )
}

function MeasureBlock({ measure, analysis, pathToUrl }) {
  const model = analysis?.model || {}
  const het = analysis?.heterogeneity || {}
  const grade = analysis?.grade || {}
  const plots = analysis?.plots || {}
  const forestUrl = pathToUrl(plots.forest)
  const funnelUrl = pathToUrl(plots.funnel)

  return (
    <div className="mt-3 p-4 rounded-xl border border-sky-100 bg-sky-50/30">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-xs font-semibold text-sky-700 uppercase">{measure}</span>
        <span className="text-xs text-zinc-400">({model.k_studies ?? '?'} studies)</span>
      </div>

      <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2">
        <MetricCell label="Pooled" value={model.pooled_estimate} />
        <MetricCell
          label="95% CI"
          value={
            model.ci_lower !== undefined
              ? `[${model.ci_lower}, ${model.ci_upper}]`
              : '—'
          }
        />
        <MetricCell label="I²" value={het.I2 !== undefined ? `${het.I2}%` : '—'} />
        <MetricCell label="GRADE" value={grade.final_level || '—'} />
      </div>

      {het.interpretation && (
        <p className="mt-2 text-xs text-zinc-500">
          p = {model.p_value ?? '?'} · {het.interpretation}
        </p>
      )}

      {forestUrl && (
        <div className="mt-4">
          <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wide mb-2">
            Forest Plot
          </p>
          <img
            src={forestUrl}
            alt={`Forest plot ${measure}`}
            className="w-full rounded-lg border border-zinc-200 bg-white"
          />
        </div>
      )}

      {funnelUrl && (
        <details className="mt-3 group">
          <summary className="cursor-pointer text-xs font-medium text-zinc-500 hover:text-zinc-700 flex items-center gap-1">
            <ChevronRight size={12} className="group-open:rotate-90 transition-transform" />
            Funnel plot
          </summary>
          <img
            src={funnelUrl}
            alt={`Funnel plot ${measure}`}
            className="mt-2 w-full rounded-lg border border-zinc-200 bg-white"
          />
        </details>
      )}

      {grade?.domains && Object.keys(grade.domains).length > 0 && (
        <details className="mt-3 group">
          <summary className="cursor-pointer text-xs font-medium text-zinc-500 hover:text-zinc-700 flex items-center gap-1">
            <ChevronRight size={12} className="group-open:rotate-90 transition-transform" />
            GRADE details
          </summary>
          <div className="mt-2 text-xs text-zinc-600 space-y-1">
            <div>
              <span className="font-semibold">Starting:</span> {grade.starting_level ?? '?'} ·{' '}
              <span className="font-semibold">Final:</span> {grade.final_level ?? '?'} ·{' '}
              <span className="font-semibold">Downgrades:</span> {grade.total_downgrades ?? 0}
            </div>
            {Object.entries(grade.domains).map(([d, info]) => {
              const dg = info.downgrades ?? info.downgrade ?? 0
              return (
                <div key={d} className="pl-3">
                  <span className="font-medium">{d.replace(/_/g, ' ')}:</span> {info.reason || '—'}
                  {dg > 0 && <span className="text-rose-600 font-semibold"> (−{dg})</span>}
                </div>
              )
            })}
          </div>
        </details>
      )}
    </div>
  )
}

function MetricCell({ label, value }) {
  return (
    <div className="p-2.5 rounded-lg bg-white border border-sky-100">
      <div className="text-[10px] text-zinc-400 uppercase tracking-wide">{label}</div>
      <div className="font-mono text-sm font-bold text-zinc-900 mt-0.5">{value ?? '—'}</div>
    </div>
  )
}

// ── A5 Studies Table ─────────────────────────────────────────────────────────

function StudiesSection({ papers }) {
  return (
    <Accordion
      icon={<Table2 size={16} className="text-violet-600" />}
      title="A5 · Estudios incluidos"
      subtitle={`${papers.length} ${papers.length === 1 ? 'estudio' : 'estudios'} con extracción completa`}
      defaultOpen={false}
    >
      <StudiesTable papers={papers} />
    </Accordion>
  )
}

// ── PRISMA ───────────────────────────────────────────────────────────────────

function PRISMASection({ prisma, pathToUrl }) {
  const diagram = pathToUrl(prisma?.prisma_flow_diagram)
  if (!diagram) return null
  return (
    <Accordion
      icon={<FileText size={16} className="text-zinc-600" />}
      title="PRISMA 2020 Flow Diagram"
      subtitle={`${prisma.included_final ?? 0} included of ${prisma.identified ?? 0} identified`}
    >
      <img
        src={diagram}
        alt="PRISMA Flow Diagram"
        className="w-full rounded-lg border border-zinc-200 bg-white"
      />
    </Accordion>
  )
}

// ── A6 Quality ───────────────────────────────────────────────────────────────

function A6Section({ a6, pathToUrl }) {
  const papers = a6?.papers || []
  const stats = a6?.stats || {}
  const tlUrl = pathToUrl(a6?.rob_traffic_light_plot)
  const sumUrl = pathToUrl(a6?.rob_summary_plot)

  const icon = (overall) => {
    const good = ['Low', 'Include'].includes(overall)
    const bad = ['High', 'Serious', 'Critical', 'Exclude'].includes(overall)
    if (good) return '✓'
    if (bad) return '✗'
    return '⚠'
  }

  return (
    <Accordion
      icon={<ShieldCheck size={16} className="text-emerald-600" />}
      title="A6 · Quality Assessment"
      subtitle={`${papers.length} papers · RoB2:${stats.rob2_assessed || 0} · ROBINS-I:${stats.robins_i_assessed || 0} · JBI:${stats.jbi_assessed || 0}`}
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        {tlUrl && (
          <div>
            <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wide mb-1.5">
              Traffic Light
            </p>
            <img src={tlUrl} alt="RoB Traffic Light" className="w-full rounded-lg border border-zinc-200 bg-white" />
          </div>
        )}
        {sumUrl && (
          <div>
            <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wide mb-1.5">
              Summary
            </p>
            <img src={sumUrl} alt="RoB Summary" className="w-full rounded-lg border border-zinc-200 bg-white" />
          </div>
        )}
      </div>

      <div className="space-y-1.5 max-h-96 overflow-y-auto">
        {papers.map((p, i) => {
          const qa = p.quality_assessment || {}
          const tool = qa.tool || '?'
          const overall = qa.overall || '?'
          const weight = p.quality_weight ?? '—'
          return (
            <details key={p.pmid || i} className="group p-2.5 rounded-lg border border-zinc-100 bg-white hover:border-emerald-200 transition-colors">
              <summary className="cursor-pointer flex items-start gap-2 text-sm">
                <span className="font-mono text-base shrink-0">{icon(overall)}</span>
                <div className="flex-1 min-w-0">
                  <div className="truncate text-zinc-700">
                    [{p.pmid || '?'}] {(p.title || '').slice(0, 100)}
                  </div>
                  <div className="text-[11px] text-zinc-400 mt-0.5">
                    {tool} · {overall} · weight={weight}
                  </div>
                </div>
              </summary>
              <div className="mt-2 pl-6 text-xs text-zinc-600 space-y-1">
                {qa.overall_justification && <p className="italic">{qa.overall_justification}</p>}
                {Object.entries(qa.domains || qa.items || {}).map(([d, info]) => (
                  <div key={d}>
                    <span className="font-semibold">{d.replace(/_/g, ' ')}:</span>{' '}
                    {info.judgment || info.answer || '?'}
                    {info.justification && (
                      <span className="text-zinc-500"> — {info.justification}</span>
                    )}
                  </div>
                ))}
              </div>
            </details>
          )
        })}
      </div>
    </Accordion>
  )
}

// ── A8 Manuscript ────────────────────────────────────────────────────────────

const SECTION_LABELS = {
  abstract: 'Abstract',
  introduction: 'Introduction',
  methods: 'Methods',
  results: 'Results',
  discussion: 'Discussion',
  conclusion: 'Conclusion',
}

function A8Section({ a8, pathToUrl }) {
  const manuscript = a8?.manuscript || {}
  const references = a8?.references || []
  const stats = a8?.stats || {}
  const pdfUrl = pathToUrl(a8?.pdf_path)
  const [activeTab, setActiveTab] = useState('abstract')

  const activeText = manuscript[activeTab] || ''
  const wordCount = activeText.trim() ? activeText.trim().split(/\s+/).length : 0

  return (
    <Accordion
      icon={<FileOutput size={16} className="text-zinc-600" />}
      title="A8 · Manuscript Draft"
      subtitle={`${stats.total_papers_cited || 0} citations · ${stats.total_chunks || 0} RAG chunks`}
    >
      <div className="flex flex-wrap gap-1.5 mb-3 border-b border-zinc-100 pb-2">
        {Object.keys(SECTION_LABELS).map((key) => {
          const has = !!manuscript[key]
          return (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              disabled={!has}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                activeTab === key
                  ? 'bg-zinc-900 text-white'
                  : has
                  ? 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200'
                  : 'bg-zinc-50 text-zinc-300 cursor-not-allowed'
              }`}
            >
              {SECTION_LABELS[key]}
            </button>
          )
        })}
      </div>

      <div className="text-xs text-zinc-400 mb-2">{wordCount} words</div>
      <div className="prose prose-sm max-w-none text-zinc-700 whitespace-pre-wrap leading-relaxed max-h-80 overflow-y-auto">
        {activeText || <em className="text-zinc-400">Not generated.</em>}
      </div>

      {references.length > 0 && (
        <details className="mt-4 group">
          <summary className="cursor-pointer text-xs font-medium text-zinc-500 hover:text-zinc-700 flex items-center gap-1">
            <ChevronRight size={12} className="group-open:rotate-90 transition-transform" />
            References ({references.length})
          </summary>
          <ol className="mt-2 pl-5 text-xs text-zinc-600 space-y-1 list-decimal max-h-60 overflow-y-auto">
            {references.map((ref, i) => (
              <li key={i}>
                {ref.citation || '?'}. <em>{ref.title || '?'}</em>.{' '}
                {ref.journal || '?'}. PMID: {ref.pmid || '?'}
              </li>
            ))}
          </ol>
        </details>
      )}

      {pdfUrl && (
        <a
          href={pdfUrl}
          download
          className="mt-4 inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-700 transition-colors"
        >
          <Download size={14} /> Manuscript PDF
        </a>
      )}
    </Accordion>
  )
}

// ── Protocol (A1) ────────────────────────────────────────────────────────────

function ProtocolSection({ protocol }) {
  const fw = protocol?.framework || '?'
  const inc = protocol?.inclusion_criteria || []
  const exc = protocol?.exclusion_criteria || []
  return (
    <Accordion
      icon={<FileText size={16} className="text-sam-600" />}
      title="A1 · Protocol"
      subtitle={fw}
    >
      <dl className="text-sm space-y-1.5 text-zinc-700">
        <Field label="Framework" value={fw} />
        <Field label="Population" value={protocol.population} />
        {fw === 'PECO' ? (
          <Field label="Exposure" value={protocol.exposure || protocol.intervention} />
        ) : fw === 'SPIDER' ? (
          <Field label="Phenomenon" value={protocol.phenomenon_of_interest || protocol.intervention} />
        ) : (
          <Field label="Intervention" value={protocol.intervention} />
        )}
        <Field label="Comparator" value={protocol.comparator} />
        <Field label="Outcome" value={protocol.outcome} />
      </dl>
      {inc.length > 0 && (
        <div className="mt-3">
          <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wide mb-1">
            Inclusion
          </p>
          <ul className="text-xs text-zinc-600 list-disc pl-5 space-y-0.5">
            {inc.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      )}
      {exc.length > 0 && (
        <div className="mt-3">
          <p className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wide mb-1">
            Exclusion
          </p>
          <ul className="text-xs text-zinc-600 list-disc pl-5 space-y-0.5">
            {exc.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      )}
    </Accordion>
  )
}

function Field({ label, value }) {
  return (
    <div className="flex gap-2">
      <dt className="text-zinc-400 text-xs font-semibold uppercase tracking-wide shrink-0 w-24">
        {label}
      </dt>
      <dd className="text-zinc-700">{value || '—'}</dd>
    </div>
  )
}

// ── Shared Accordion ─────────────────────────────────────────────────────────

function Accordion({ icon, title, subtitle, children, defaultOpen = false }) {
  return (
    <details
      className="group rounded-xl border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900/60 overflow-hidden"
      open={defaultOpen}
    >
      <summary className="cursor-pointer p-4 flex items-center gap-3 hover:bg-zinc-50 dark:hover:bg-zinc-800/60 transition-colors list-none">
        <ChevronRight
          size={16}
          className="text-zinc-400 dark:text-zinc-500 group-open:rotate-90 transition-transform shrink-0"
        />
        {icon}
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">{title}</h3>
          {subtitle && <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-0.5">{subtitle}</p>}
        </div>
      </summary>
      <div className="p-4 border-t border-zinc-100 dark:border-zinc-800">
        {children}
      </div>
    </details>
  )
}
