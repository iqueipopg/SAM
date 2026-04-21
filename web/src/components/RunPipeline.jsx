import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Play, Square, ChevronDown, ChevronUp, Loader2, Check, X,
  AlertTriangle, Download, Settings2, ArrowRight,
  FileText, Search, Filter, BookOpen, Database,
  ShieldCheck, BarChart3, FileOutput, ScrollText
} from 'lucide-react'
import ResultsPanel from './ResultsPanel'

const API_URL = import.meta.env.VITE_API_URL || ''

const EXAMPLE_QUERIES = [
  {
    label: 'CBT vs medication for depression in adults',
    query: 'Is cognitive behavioral therapy more effective than medication for treating depression in adults?',
  },
  {
    label: 'Mindfulness for anxiety in adolescents',
    query: 'Is mindfulness meditation effective for reducing anxiety in adolescents?',
  },
  {
    label: 'Exercise and cognition in dementia',
    query: 'Does exercise improve cognitive function in elderly patients with dementia?',
  },
  {
    label: 'Smoking and MI risk in diabetes',
    query: 'Does smoking increase myocardial infarction risk in diabetic patients?',
  },
  {
    label: 'Barriers to evidence-based nursing',
    query: 'What barriers do nurses perceive in implementing evidence-based practice?',
  },
]

const AGENT_META = {
  A1: { name: 'Protocol', icon: FileText, color: 'text-sam-600 bg-sam-50 border-sam-200' },
  A2: { name: 'Search', icon: Search, color: 'text-blue-600 bg-blue-50 border-blue-200' },
  A3: { name: 'Screening', icon: Filter, color: 'text-amber-600 bg-amber-50 border-amber-200' },
  A4: { name: 'Full-text', icon: BookOpen, color: 'text-rose-600 bg-rose-50 border-rose-200' },
  A5: { name: 'Extraction', icon: Database, color: 'text-violet-600 bg-violet-50 border-violet-200' },
  A6: { name: 'Quality', icon: ShieldCheck, color: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
  A7: { name: 'Synthesis', icon: BarChart3, color: 'text-sky-600 bg-sky-50 border-sky-200' },
  A8: { name: 'Manuscript', icon: FileOutput, color: 'text-zinc-600 bg-zinc-50 border-zinc-200' },
  A9: { name: 'Explainer', icon: ScrollText, color: 'text-teal-600 bg-teal-50 border-teal-200' },
}

const AGENTS_ORDER = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']

export default function RunPipeline() {
  const [query, setQuery] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState({
    max_results: 40,
    dedup_threshold: 0.90,
    a3_ensemble: 3,
    a3_vote_threshold: 0.60,
    a4_ensemble: 3,
    a4_vote_threshold: 0.60,
  })

  const [running, setRunning] = useState(false)
  const [agents, setAgents] = useState({}) // { A1: { status, summary, error } }
  const [currentAgent, setCurrentAgent] = useState(null)
  const [pipelineResult, setPipelineResult] = useState(null)
  const [pipelineError, setPipelineError] = useState(null)
  const [downloads, setDownloads] = useState({})
  const [runId, setRunId] = useState(null)
  const [fullData, setFullData] = useState(null)
  const [loadingResults, setLoadingResults] = useState(false)

  const logRef = useRef(null)
  const readerRef = useRef(null)
  const abortRef = useRef(null)

  const scrollLog = useCallback(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [])

  useEffect(scrollLog, [agents, currentAgent, scrollLog])

  const startPipeline = async () => {
    if (!query.trim()) return

    setRunning(true)
    setAgents({})
    setCurrentAgent(null)
    setPipelineResult(null)
    setPipelineError(null)
    setDownloads({})
    setRunId(null)
    setFullData(null)

    try {
      const controller = new AbortController()
      abortRef.current = controller

      const response = await fetch(`${API_URL}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), ...settings }),
        signal: controller.signal,
      })

      const reader = response.body.getReader()
      readerRef.current = reader
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let eventType = null
        let eventData = null

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            try {
              eventData = JSON.parse(line.slice(6))
            } catch {
              continue
            }

            if (eventType && eventData) {
              handleSSE(eventType, eventData)
              eventType = null
              eventData = null
            }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setPipelineError('Pipeline killed by user')
      } else {
        setPipelineError(`Connection error: ${err.message}`)
      }
    } finally {
      setRunning(false)
      setCurrentAgent(null)
      readerRef.current = null
      abortRef.current = null
    }
  }

  const handleSSE = (event, data) => {
    switch (event) {
      case 'start':
        setRunId(data.run_dir?.split('/').pop())
        break

      case 'agent_start':
        setCurrentAgent(data.agent)
        setAgents(prev => ({
          ...prev,
          [data.agent]: { status: 'running' }
        }))
        break

      case 'agent_done':
        setAgents(prev => ({
          ...prev,
          [data.agent]: { status: 'done', summary: data.summary }
        }))
        setCurrentAgent(null)
        break

      case 'agent_error':
        setAgents(prev => ({
          ...prev,
          [data.agent]: { status: 'error', error: data.error }
        }))
        break

      case 'agent_warning':
        setAgents(prev => ({
          ...prev,
          [data.agent]: { status: 'warning', warning: data.warning }
        }))
        break

      case 'pipeline_done': {
        setPipelineResult(data)
        setDownloads(data.downloads || {})
        const rid = data.run_dir?.split('/').pop()
        setRunId(rid)
        if (rid) {
          setLoadingResults(true)
          fetch(`${API_URL}/api/download/${rid}/pipeline.json`)
            .then((r) => (r.ok ? r.json() : null))
            .then((json) => setFullData(json))
            .catch(() => setFullData(null))
            .finally(() => setLoadingResults(false))
        }
        break
      }

      case 'pipeline_error':
        setPipelineError(data.error)
        break

      case 'pipeline_stopped':
        setPipelineError(`Pipeline stopped after ${data.last_agent}`)
        break

      case 'finalizing':
        break
    }
  }

  const stopPipeline = async () => {
    try {
      await fetch(`${API_URL}/api/stop`, { method: 'POST' })
    } catch {}
  }

  const killPipeline = async () => {
    // 1. Tell backend to kill immediately
    try {
      await fetch(`${API_URL}/api/kill`, { method: 'POST' })
    } catch {}
    // 2. Abort the SSE connection client-side
    if (abortRef.current) {
      abortRef.current.abort()
    }
  }

  const selectExample = (q) => {
    if (!running) setQuery(q.query)
  }

  const updateSetting = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }))
  }

  return (
    <section id="run" className="py-24 md:py-32">
      <div className="max-w-[1400px] mx-auto px-6 md:px-10">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto mb-12">
          <span className="text-xs font-semibold tracking-[0.15em] uppercase text-sam-600">
            Run
          </span>
          <h2 className="mt-3 text-3xl md:text-4xl font-800 tracking-tighter text-zinc-950">
            Run a meta-analysis
          </h2>
          <p className="mt-4 text-zinc-500 leading-relaxed">
            Enter a research question or select an example. SAM will execute the
            full 9-agent pipeline and produce downloadable outputs.
          </p>
        </div>

        <div className="max-w-[1100px] mx-auto">
          {/* Query input */}
          <div className="rounded-2xl bg-white border border-zinc-200 shadow-[0_4px_16px_rgba(0,0,0,0.04)] overflow-hidden">
            <div className="p-6">
              <label className="block text-xs font-semibold tracking-[0.1em] uppercase text-zinc-400 mb-3">
                Research question
              </label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., Is cognitive behavioral therapy more effective than medication for treating depression in adults?"
                rows={3}
                disabled={running}
                className="w-full px-4 py-3 rounded-xl border border-zinc-200 bg-[#FAFAFA] text-zinc-900 text-sm leading-relaxed placeholder:text-zinc-400 focus:outline-none focus:border-sam-400 focus:ring-2 focus:ring-sam-100 transition-all resize-none disabled:opacity-60"
              />

              {/* Example chips */}
              <div className="mt-4 flex flex-wrap gap-2">
                {EXAMPLE_QUERIES.map((q) => (
                  <button
                    key={q.label}
                    onClick={() => selectExample(q)}
                    disabled={running}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                      query === q.query
                        ? 'bg-sam-50 text-sam-700 border border-sam-200'
                        : 'bg-zinc-50 text-zinc-500 border border-zinc-100 hover:border-zinc-200 hover:text-zinc-700'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {q.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Settings toggle */}
            <div className="border-t border-zinc-100">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="w-full px-6 py-3 flex items-center justify-between text-xs font-medium text-zinc-400 hover:text-zinc-600 transition-colors"
              >
                <span className="flex items-center gap-2">
                  <Settings2 size={14} />
                  Pipeline settings
                </span>
                {showSettings ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>

              {showSettings && (
                <div className="px-6 pb-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  <SettingSlider
                    label="Max papers (total target)"
                    value={settings.max_results}
                    min={40} max={200} step={10}
                    onChange={(v) => updateSetting('max_results', v)}
                    tooltip="Con 4 bases de datos y floor 10 papers/DB, valores < 40 son subóptimos."
                  />
                  <SettingSlider
                    label="Dedup threshold"
                    value={settings.dedup_threshold}
                    min={0.80} max={0.99} step={0.01}
                    onChange={(v) => updateSetting('dedup_threshold', v)}
                    format={(v) => v.toFixed(2)}
                  />
                  <SettingSelect
                    label="A3 Ensemble size"
                    value={settings.a3_ensemble}
                    options={[1, 3, 5]}
                    onChange={(v) => updateSetting('a3_ensemble', Number(v))}
                  />
                  <SettingSlider
                    label="A3 Vote threshold"
                    value={settings.a3_vote_threshold}
                    min={0.50} max={0.90} step={0.05}
                    onChange={(v) => updateSetting('a3_vote_threshold', v)}
                    format={(v) => v.toFixed(2)}
                  />
                  <SettingSelect
                    label="A4 Ensemble size"
                    value={settings.a4_ensemble}
                    options={[1, 3, 5]}
                    onChange={(v) => updateSetting('a4_ensemble', Number(v))}
                  />
                  <SettingSlider
                    label="A4 Vote threshold"
                    value={settings.a4_vote_threshold}
                    min={0.50} max={0.90} step={0.05}
                    onChange={(v) => updateSetting('a4_vote_threshold', v)}
                    format={(v) => v.toFixed(2)}
                  />
                </div>
              )}
            </div>

            {/* Action bar */}
            <div className="px-6 py-4 bg-zinc-50 border-t border-zinc-100 flex items-center justify-between">
              <p className="text-xs text-zinc-400">
                {running ? 'Pipeline is running...' : 'Ready to run'}
              </p>
              <div className="flex gap-3">
                {running && (
                  <>
                    <button
                      onClick={stopPipeline}
                      className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-zinc-200 text-zinc-700 text-sm font-medium hover:bg-zinc-300 transition-all active:scale-[0.98]"
                    >
                      <Square size={14} fill="currentColor" />
                      Stop
                    </button>
                    <button
                      onClick={killPipeline}
                      className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-red-100 text-red-700 text-sm font-medium hover:bg-red-200 transition-all active:scale-[0.98]"
                    >
                      <X size={14} strokeWidth={3} />
                      Kill
                    </button>
                  </>
                )}
                <button
                  onClick={startPipeline}
                  disabled={running || !query.trim()}
                  className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-sam-600 text-white text-sm font-medium hover:bg-sam-700 transition-all hover:-translate-y-[1px] active:translate-y-0 shadow-[0_4px_12px_rgba(13,148,136,0.3)] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:shadow-none"
                >
                  <Play size={14} fill="currentColor" />
                  Run pipeline
                </button>
              </div>
            </div>
          </div>

          {/* Pipeline progress */}
          {(running || Object.keys(agents).length > 0) && (
            <div className="mt-8 rounded-2xl bg-white border border-zinc-200 shadow-[0_4px_16px_rgba(0,0,0,0.04)] overflow-hidden">
              {/* Agent timeline */}
              <div ref={logRef} className="p-6 max-h-[500px] overflow-y-auto">
                <p className="text-xs font-semibold tracking-[0.1em] uppercase text-zinc-400 mb-5">
                  Pipeline progress
                </p>

                <div className="space-y-3">
                  {AGENTS_ORDER.map((agentId) => {
                    const meta = AGENT_META[agentId]
                    const state = agents[agentId]
                    const Icon = meta.icon
                    const isActive = currentAgent === agentId
                    const isPending = !state && !isActive

                    return (
                      <div
                        key={agentId}
                        className={`flex items-start gap-4 p-4 rounded-xl border transition-all duration-300 ${
                          isActive
                            ? 'border-sam-200 bg-sam-50/30 shadow-[0_2px_8px_rgba(13,148,136,0.06)]'
                            : state?.status === 'done'
                            ? 'border-emerald-100 bg-emerald-50/30'
                            : state?.status === 'error'
                            ? 'border-red-100 bg-red-50/30'
                            : state?.status === 'warning'
                            ? 'border-amber-100 bg-amber-50/30'
                            : 'border-zinc-100 bg-zinc-50/30 opacity-40'
                        }`}
                      >
                        {/* Status icon */}
                        <div className={`w-9 h-9 rounded-lg border flex items-center justify-center flex-shrink-0 ${meta.color}`}>
                          <Icon size={16} />
                        </div>

                        {/* Content */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs font-semibold text-zinc-400">{agentId}</span>
                            <span className="text-sm font-medium text-zinc-800">{meta.name}</span>
                            {isActive && (
                              <Loader2 size={14} className="text-sam-500 animate-spin" />
                            )}
                            {state?.status === 'done' && (
                              <Check size={14} className="text-emerald-500" />
                            )}
                            {state?.status === 'error' && (
                              <X size={14} className="text-red-500" />
                            )}
                            {state?.status === 'warning' && (
                              <AlertTriangle size={14} className="text-amber-500" />
                            )}
                          </div>

                          {/* Summary */}
                          {state?.summary && (
                            <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
                              {Object.entries(state.summary).map(([k, v]) => (
                                <span key={k} className="text-xs text-zinc-500">
                                  <span className="text-zinc-400">{k.replace(/_/g, ' ')}:</span>{' '}
                                  <span className="font-mono font-medium text-zinc-700">
                                    {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                                  </span>
                                </span>
                              ))}
                            </div>
                          )}

                          {state?.error && (
                            <p className="mt-1 text-xs text-red-600">{state.error}</p>
                          )}
                          {state?.warning && (
                            <p className="mt-1 text-xs text-amber-600">{state.warning}</p>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Error */}
                {pipelineError && (
                  <div className="mt-4 p-4 rounded-xl bg-red-50 border border-red-100">
                    <p className="text-sm text-red-700 font-medium">{pipelineError}</p>
                  </div>
                )}
              </div>

              {/* Downloads bar */}
              {pipelineResult && Object.keys(downloads).length > 0 && (
                <div className="px-6 py-5 border-t border-zinc-100 bg-zinc-50">
                  <p className="text-xs font-semibold tracking-[0.1em] uppercase text-zinc-400 mb-4">
                    Downloads
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(downloads).map(([label, filepath]) => (
                      <a
                        key={label}
                        href={`${API_URL}/api/download/${runId}/${filepath}`}
                        download
                        className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white border border-zinc-200 text-sm font-medium text-zinc-700 hover:border-sam-300 hover:text-sam-700 hover:bg-sam-50/50 transition-all duration-200 hover:-translate-y-[1px] active:translate-y-0 shadow-[0_1px_3px_rgba(0,0,0,0.04)]"
                      >
                        <Download size={14} />
                        {label}
                      </a>
                    ))}
                  </div>

                  {/* PRISMA summary */}
                  {pipelineResult.prisma && (
                    <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
                      {[
                        { label: 'Identified', value: pipelineResult.prisma.identified },
                        { label: 'After dedup', value: pipelineResult.prisma.after_dedup },
                        { label: 'Screened', value: pipelineResult.prisma.screened },
                        { label: 'Full-text', value: pipelineResult.prisma.assessed_fulltext },
                        { label: 'Included', value: pipelineResult.prisma.included_final },
                        { label: 'Quality assessed', value: pipelineResult.prisma.quality_assessed },
                      ].map(({ label, value }) => (
                        <div key={label} className="text-center p-3 rounded-lg bg-white border border-zinc-100">
                          <div className="font-mono text-lg font-bold text-zinc-900">{value ?? 0}</div>
                          <div className="text-[11px] text-zinc-400 mt-0.5">{label}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Results panel (A9, A7, A6, A8, PRISMA, A1) */}
              {pipelineResult && loadingResults && (
                <div className="px-6 py-5 border-t border-zinc-100 flex items-center gap-2 text-sm text-zinc-500">
                  <Loader2 size={14} className="animate-spin" />
                  Loading detailed results...
                </div>
              )}
              {pipelineResult && fullData && (
                <ResultsPanel fullData={fullData} runId={runId} apiUrl={API_URL} />
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}


// ── Sub-components ──────────────────────────────────────────────────────────

function SettingSlider({ label, value, min, max, step, onChange, format, tooltip }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label
          className="text-xs text-zinc-500 font-medium"
          title={tooltip || undefined}
        >
          {label}
          {tooltip && (
            <span
              className="ml-1 text-zinc-400 cursor-help"
              title={tooltip}
              aria-label={tooltip}
            >
              &#9432;
            </span>
          )}
        </label>
        <span className="font-mono text-xs text-zinc-700 font-semibold">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full bg-zinc-200 appearance-none cursor-pointer accent-sam-600"
      />
      {tooltip && (
        <p className="mt-1 text-[10px] text-zinc-400 leading-snug">{tooltip}</p>
      )}
    </div>
  )
}

function SettingSelect({ label, value, options, onChange }) {
  return (
    <div>
      <label className="block text-xs text-zinc-500 font-medium mb-1.5">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 rounded-lg border border-zinc-200 bg-white text-sm text-zinc-700 focus:outline-none focus:border-sam-400"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    </div>
  )
}
