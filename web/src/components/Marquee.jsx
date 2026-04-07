export default function Marquee() {
  const items = [
    'PubMed', 'Europe PMC', 'Semantic Scholar', 'Google Scholar',
    'RoB 2.0', 'ROBINS-I', 'JBI', 'GRADE', 'PRISMA 2020',
    'R/metafor', 'Forest plots', 'Funnel plots', 'Leave-one-out',
    'HuggingFace', 'FAISS', 'LaTeX PDF',
  ]

  return (
    <div className="relative py-6 border-y border-zinc-100 bg-white overflow-hidden">
      <div className="animate-marquee whitespace-nowrap flex gap-10">
        {[...items, ...items].map((item, i) => (
          <span
            key={i}
            className="text-sm font-medium text-zinc-400 tracking-wide uppercase"
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  )
}
