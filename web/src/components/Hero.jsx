import { useEffect, useRef } from 'react'
import gsap from 'gsap'
import { ArrowRight, Zap, Clock, ShieldCheck } from 'lucide-react'

export default function Hero() {
  const containerRef = useRef(null)
  const titleRef = useRef(null)
  const subtitleRef = useRef(null)
  const ctaRef = useRef(null)
  const badgesRef = useRef(null)
  const visualRef = useRef(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } })

      tl.fromTo(titleRef.current, { y: 40, opacity: 0 }, { y: 0, opacity: 1, duration: 1 })
        .fromTo(subtitleRef.current, { y: 30, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8 }, '-=0.6')
        .fromTo(ctaRef.current, { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.6 }, '-=0.4')
        .fromTo(badgesRef.current, { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.6 }, '-=0.3')
        .fromTo(visualRef.current, { y: 40, opacity: 0, scale: 0.97 }, { y: 0, opacity: 1, scale: 1, duration: 1 }, '-=0.5')
    }, containerRef)

    return () => ctx.revert()
  }, [])

  return (
    <section ref={containerRef} className="relative min-h-[100dvh] flex items-center pt-24 pb-16 overflow-hidden">
      {/* Mesh gradient background */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="mesh-blob absolute -top-32 -left-32 w-[500px] h-[500px] rounded-full bg-sam-200/30 blur-[100px]" />
        <div className="mesh-blob-delay absolute top-1/3 right-0 w-[400px] h-[400px] rounded-full bg-sam-100/40 blur-[80px]" />
        <div className="mesh-blob-delay-2 absolute bottom-0 left-1/3 w-[350px] h-[350px] rounded-full bg-teal-50/50 blur-[90px]" />
      </div>

      <div className="max-w-[1400px] mx-auto px-6 md:px-10 w-full relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-20 items-center">
          {/* Left — Content */}
          <div>
            <h1
              ref={titleRef}
              className="text-4xl sm:text-5xl md:text-6xl font-800 tracking-tighter leading-[1.05] text-zinc-950 opacity-0"
            >
              From research question
              <br />
              to published manuscript.
              <br />
              <span className="text-sam-600">Automated.</span>
            </h1>

            <p
              ref={subtitleRef}
              className="mt-6 text-lg text-zinc-500 leading-relaxed max-w-[52ch] opacity-0"
            >
              SAM orchestrates 8 specialized AI agents to execute the full cycle
              of a systematic review and meta-analysis — protocol design, literature
              search, screening, data extraction, quality assessment, statistical
              synthesis, and manuscript generation.
            </p>

            <div ref={ctaRef} className="mt-8 flex flex-wrap gap-4 opacity-0">
              <a
                href="#run"
                className="inline-flex items-center gap-2.5 px-6 py-3.5 rounded-full bg-sam-600 text-white font-medium text-sm hover:bg-sam-700 transition-all duration-200 hover:-translate-y-[1px] active:translate-y-0 shadow-[0_4px_16px_rgba(13,148,136,0.3)]"
              >
                Try a query
                <ArrowRight size={16} strokeWidth={2.5} />
              </a>
              <a
                href="#pipeline"
                className="inline-flex items-center gap-2 px-6 py-3.5 rounded-full border border-zinc-200 bg-white text-zinc-700 font-medium text-sm hover:border-zinc-300 hover:bg-zinc-50 transition-all duration-200 hover:-translate-y-[1px] active:translate-y-0"
              >
                Explore pipeline
              </a>
            </div>

            <div ref={badgesRef} className="mt-10 flex flex-wrap gap-3 opacity-0">
              {[
                { icon: Clock, text: '<30 min per review' },
                { icon: ShieldCheck, text: 'PRISMA & GRADE compliant' },
                { icon: Zap, text: 'Open source' },
              ].map(({ icon: Icon, text }) => (
                <div
                  key={text}
                  className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white border border-zinc-100 text-xs text-zinc-500 font-medium shadow-[0_1px_3px_rgba(0,0,0,0.04)]"
                >
                  <Icon size={14} strokeWidth={2} className="text-sam-600" />
                  {text}
                </div>
              ))}
            </div>
          </div>

          {/* Right — Visual */}
          <div ref={visualRef} className="opacity-0 hidden lg:block">
            <div className="relative">
              {/* Agent cards floating */}
              <div className="relative w-full aspect-square max-w-[480px] ml-auto">
                {/* Central node */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-20 h-20 rounded-2xl bg-sam-600 shadow-[0_8px_32px_rgba(13,148,136,0.4)] flex items-center justify-center z-10">
                  <span className="text-white font-bold text-xl tracking-tighter">SAM</span>
                </div>

                {/* Orbiting agents */}
                {[
                  { label: 'A1', sub: 'Protocol', x: '50%', y: '5%', delay: '' },
                  { label: 'A2', sub: 'Search', x: '85%', y: '20%', delay: 'animate-float-delay' },
                  { label: 'A3', sub: 'Screen', x: '95%', y: '50%', delay: 'animate-float-delay-2' },
                  { label: 'A4', sub: 'Full-text', x: '85%', y: '78%', delay: '' },
                  { label: 'A5', sub: 'Extract', x: '50%', y: '93%', delay: 'animate-float-delay' },
                  { label: 'A6', sub: 'Quality', x: '15%', y: '78%', delay: 'animate-float-delay-2' },
                  { label: 'A7', sub: 'Synthesis', x: '5%', y: '50%', delay: '' },
                  { label: 'A8', sub: 'Manuscript', x: '15%', y: '20%', delay: 'animate-float-delay' },
                ].map(({ label, sub, x, y, delay }) => (
                  <div
                    key={label}
                    className={`absolute -translate-x-1/2 -translate-y-1/2 animate-float ${delay}`}
                    style={{ left: x, top: y }}
                  >
                    <div className="w-16 h-16 rounded-xl bg-white border border-zinc-100 shadow-[0_4px_16px_rgba(0,0,0,0.06)] flex flex-col items-center justify-center gap-0.5 hover:shadow-[0_8px_24px_rgba(0,0,0,0.1)] hover:-translate-y-1 transition-all duration-300">
                      <span className="text-sam-600 font-bold text-xs font-mono">{label}</span>
                      <span className="text-zinc-400 text-[10px] font-medium">{sub}</span>
                    </div>
                  </div>
                ))}

                {/* Connection lines — SVG */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 480 480">
                  <circle cx="240" cy="240" r="160" fill="none" stroke="#E4E4E7" strokeWidth="1" strokeDasharray="4 6" opacity="0.6" />
                  <circle cx="240" cy="240" r="100" fill="none" stroke="#E4E4E7" strokeWidth="1" strokeDasharray="4 6" opacity="0.3" />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
