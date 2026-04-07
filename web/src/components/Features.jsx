import {
  Cpu, Globe, Lock, Gauge,
  GitBranch, FileBarChart
} from 'lucide-react'

const features = [
  {
    icon: Cpu,
    title: 'Model-agnostic',
    desc: 'Works with any HuggingFace model. Ships with Qwen2.5-7B (4-bit) but swap in LLaMA, Mistral, or any compatible model.',
  },
  {
    icon: Globe,
    title: 'Multi-source retrieval',
    desc: '10-source cascade for full-text acquisition. PubMed, Unpaywall, Europe PMC, Semantic Scholar, OpenAlex, CrossRef, and more.',
  },
  {
    icon: Lock,
    title: 'Rigorous statistics',
    desc: 'All numerical analysis runs through R/metafor. Zero LLM dependency for calculations. REML, heterogeneity, Egger, GRADE.',
  },
  {
    icon: Gauge,
    title: 'Single GPU, 8GB VRAM',
    desc: 'Runs on an RTX 4060. Agents load and unload models sequentially with free_vram(), keeping memory under 8GB at all times.',
  },
  {
    icon: GitBranch,
    title: 'Open source',
    desc: 'MIT licensed. Full pipeline, experiments, tests, and deployment scripts. Fork it, extend it, contribute back.',
  },
  {
    icon: FileBarChart,
    title: 'Complete outputs',
    desc: 'PRISMA flow diagram, forest plots, funnel plots, RoB traffic lights, GRADE tables, and a publication-ready LaTeX manuscript.',
  },
]

export default function Features() {
  return (
    <section id="features" className="py-24 md:py-32 bg-white">
      <div className="max-w-[1400px] mx-auto px-6 md:px-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-16">
          {/* Left — sticky header */}
          <div className="lg:col-span-4 lg:sticky lg:top-28 lg:self-start">
            <span className="text-xs font-semibold tracking-[0.15em] uppercase text-sam-600">
              Capabilities
            </span>
            <h2 className="mt-3 text-3xl md:text-4xl font-800 tracking-tighter text-zinc-950">
              Built for real
              <br />
              research workflows
            </h2>
            <p className="mt-4 text-zinc-500 leading-relaxed max-w-[45ch]">
              Every decision in SAM is driven by methodological standards used
              in clinical research: PRISMA 2020, RoB 2.0, GRADE, and Cochrane guidelines.
            </p>
          </div>

          {/* Right — feature list */}
          <div className="lg:col-span-8">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {features.map((feature, i) => {
                const Icon = feature.icon
                return (
                  <div
                    key={feature.title}
                    className="gsap-reveal group p-6 rounded-2xl border border-zinc-100 hover:border-sam-200 bg-[#FAFAFA] hover:bg-sam-50/30 transition-all duration-300"
                  >
                    <div className="w-10 h-10 rounded-xl bg-sam-50 border border-sam-100 flex items-center justify-center mb-4 group-hover:bg-sam-100 transition-colors duration-300">
                      <Icon size={18} strokeWidth={2} className="text-sam-600" />
                    </div>
                    <h3 className="font-semibold text-zinc-900 tracking-tight">
                      {feature.title}
                    </h3>
                    <p className="mt-2 text-sm text-zinc-500 leading-relaxed">
                      {feature.desc}
                    </p>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
