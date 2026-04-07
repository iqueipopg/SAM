import { useEffect, useRef, useState } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

const stats = [
  { value: 8, suffix: '', label: 'Specialized agents' },
  { value: 4, suffix: '', label: 'Database sources' },
  { value: 10, suffix: '', label: 'Full-text sources' },
  { value: 30, suffix: 'min', label: 'Average completion' },
  { value: 44, suffix: '', label: 'Automated tests' },
  { value: 3, suffix: '', label: 'Quality tools (RoB2, ROBINS-I, JBI)' },
]

function AnimatedNumber({ value, suffix }) {
  const [current, setCurrent] = useState(0)
  const ref = useRef(null)
  const hasAnimated = useRef(false)

  useEffect(() => {
    const trigger = ScrollTrigger.create({
      trigger: ref.current,
      start: 'top 85%',
      onEnter: () => {
        if (hasAnimated.current) return
        hasAnimated.current = true
        const obj = { val: 0 }
        gsap.to(obj, {
          val: value,
          duration: 1.5,
          ease: 'power2.out',
          onUpdate: () => setCurrent(Math.round(obj.val)),
        })
      },
    })
    return () => trigger.kill()
  }, [value])

  return (
    <span ref={ref} className="tabular-nums">
      {current}
      {suffix && <span className="text-sam-500 ml-0.5">{suffix}</span>}
    </span>
  )
}

export default function Stats() {
  return (
    <section className="py-24 md:py-32 bg-white">
      <div className="max-w-[1400px] mx-auto px-6 md:px-10">
        <div className="text-center mb-16">
          <span className="text-xs font-semibold tracking-[0.15em] uppercase text-sam-600">
            By the numbers
          </span>
          <h2 className="mt-3 text-3xl md:text-4xl font-800 tracking-tighter text-zinc-950">
            Engineered for research at scale
          </h2>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8">
          {stats.map((stat, i) => (
            <div key={stat.label} className="text-center">
              <div className="text-4xl md:text-5xl font-800 tracking-tighter text-zinc-950 font-mono">
                <AnimatedNumber value={stat.value} suffix={stat.suffix} />
              </div>
              <p className="mt-2 text-sm text-zinc-500 leading-snug">
                {stat.label}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
