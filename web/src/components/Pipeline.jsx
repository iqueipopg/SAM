import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import {
  FileText, Search, Filter, BookOpen,
  Database, ShieldCheck, BarChart3, FileOutput
} from 'lucide-react'

const agents = [
  {
    id: 'A1',
    name: 'Protocol',
    icon: FileText,
    color: 'bg-sam-50 text-sam-700 border-sam-200',
    desc: 'Detects the methodological framework (PICO, SPIDER, PECO, PCC), generates inclusion/exclusion criteria, and builds the PubMed query.',
    tech: 'LLM',
  },
  {
    id: 'A2',
    name: 'Search',
    icon: Search,
    color: 'bg-blue-50 text-blue-700 border-blue-200',
    desc: 'Searches 4 databases simultaneously (PubMed, Europe PMC, Semantic Scholar, Google Scholar) with semantic deduplication.',
    tech: 'CPU',
  },
  {
    id: 'A3',
    name: 'Screening',
    icon: Filter,
    color: 'bg-amber-50 text-amber-700 border-amber-200',
    desc: 'Evaluates each paper N times via ensemble voting on title and abstract. Parse errors default to EXCLUDE.',
    tech: 'LLM',
  },
  {
    id: 'A4',
    name: 'Full-text',
    icon: BookOpen,
    color: 'bg-rose-50 text-rose-700 border-rose-200',
    desc: '10-source cascade to retrieve full texts (PMC, Unpaywall, Europe PMC, S2, OpenAlex, CrossRef, DOAJ, CORE, GScholar, Sci-Hub), then LLM screening.',
    tech: 'HTTP + LLM',
  },
  {
    id: 'A5',
    name: 'Extraction',
    icon: Database,
    color: 'bg-violet-50 text-violet-700 border-violet-200',
    desc: 'Three focused prompts per paper: study characteristics, intervention details, and outcome data. 4-pass numeric recovery.',
    tech: 'LLM',
  },
  {
    id: 'A6',
    name: 'Quality',
    icon: ShieldCheck,
    color: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    desc: 'Auto-detects study design (RCT/observational/qualitative) and applies RoB 2.0, ROBINS-I, or JBI Critical Appraisal with numeric weights.',
    tech: 'LLM',
  },
  {
    id: 'A7',
    name: 'Synthesis',
    icon: BarChart3,
    color: 'bg-sky-50 text-sky-700 border-sky-200',
    desc: 'Random-effects meta-analysis via R/metafor: REML, heterogeneity, Egger, trim-and-fill, leave-one-out, subgroups, meta-regression, GRADE.',
    tech: 'R',
  },
  {
    id: 'A8',
    name: 'Manuscript',
    icon: FileOutput,
    color: 'bg-zinc-100 text-zinc-700 border-zinc-200',
    desc: 'RAG-enhanced section-by-section generation using FAISS + MiniLM. Data-driven prose, not hallucinated. LaTeX PDF output.',
    tech: 'LLM + RAG',
  },
]

export default function Pipeline() {
  const sectionRef = useRef(null)
  const cardsRef = useRef([])

  useEffect(() => {
    const ctx = gsap.context(() => {
      cardsRef.current.forEach((card, i) => {
        if (!card) return
        gsap.fromTo(card,
          { y: 50, opacity: 0 },
          {
            y: 0,
            opacity: 1,
            duration: 0.7,
            delay: i * 0.08,
            ease: 'power3.out',
            scrollTrigger: {
              trigger: card,
              start: 'top 88%',
              toggleActions: 'play none none none',
            },
          }
        )
      })
    }, sectionRef)
    return () => ctx.revert()
  }, [])

  return (
    <section id="pipeline" ref={sectionRef} className="py-24 md:py-32">
      <div className="max-w-[1400px] mx-auto px-6 md:px-10">
        {/* Header — left aligned */}
        <div className="max-w-xl">
          <span className="text-xs font-semibold tracking-[0.15em] uppercase text-sam-600">
            Pipeline
          </span>
          <h2 className="mt-3 text-3xl md:text-4xl font-800 tracking-tighter text-zinc-950">
            8 agents, one seamless workflow
          </h2>
          <p className="mt-4 text-zinc-500 leading-relaxed max-w-[55ch]">
            Each agent is specialized for a single phase of the systematic review process.
            They execute sequentially, passing structured data to the next stage.
          </p>
        </div>

        {/* Agent grid — asymmetric */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {agents.map((agent, i) => {
            const Icon = agent.icon
            return (
              <div
                key={agent.id}
                ref={el => cardsRef.current[i] = el}
                className="group relative p-6 rounded-2xl bg-white border border-zinc-100 shadow-[0_1px_3px_rgba(0,0,0,0.04)] hover:shadow-[0_8px_24px_rgba(0,0,0,0.08)] hover:-translate-y-1 transition-all duration-300 opacity-0"
              >
                {/* Agent number */}
                <div className="flex items-center justify-between mb-4">
                  <div className={`w-10 h-10 rounded-xl ${agent.color} border flex items-center justify-center`}>
                    <Icon size={18} strokeWidth={2} />
                  </div>
                  <span className="font-mono text-xs font-semibold text-zinc-300">
                    {agent.id}
                  </span>
                </div>

                <h3 className="font-semibold text-zinc-900 tracking-tight">
                  {agent.name}
                </h3>
                <p className="mt-2 text-sm text-zinc-500 leading-relaxed">
                  {agent.desc}
                </p>

                {/* Tech badge */}
                <div className="mt-4 inline-flex items-center px-2.5 py-1 rounded-md bg-zinc-50 border border-zinc-100">
                  <span className="font-mono text-[11px] font-medium text-zinc-400">
                    {agent.tech}
                  </span>
                </div>

                {/* Connector line — right side (hidden on last in row) */}
                {i < agents.length - 1 && (
                  <div className="hidden lg:block absolute top-1/2 -right-2 w-4 h-px bg-zinc-200" />
                )}
              </div>
            )
          })}
        </div>

        {/* Flow arrow */}
        <div className="mt-8 flex items-center justify-center gap-3">
          <div className="h-px flex-1 max-w-[200px] bg-gradient-to-r from-transparent via-zinc-200 to-transparent" />
          <span className="text-xs text-zinc-400 font-medium tracking-wide">
            A1 &rarr; A2 &rarr; A3 &rarr; A4 &rarr; A5 &rarr; A6 &rarr; A7 &rarr; A8
          </span>
          <div className="h-px flex-1 max-w-[200px] bg-gradient-to-r from-transparent via-zinc-200 to-transparent" />
        </div>
      </div>
    </section>
  )
}
