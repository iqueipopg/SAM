// Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
// SPDX-License-Identifier: Apache-2.0
import { ArrowRight } from 'lucide-react'
import GithubIcon from './GithubIcon'

export default function CTA() {
  return (
    <section className="py-24 md:py-32">
      <div className="max-w-[1400px] mx-auto px-6 md:px-10">
        <div className="relative rounded-4xl bg-zinc-950 p-10 md:p-16 overflow-hidden">
          {/* Background gradient */}
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute -top-40 -right-40 w-[500px] h-[500px] rounded-full bg-sam-600/20 blur-[100px]" />
            <div className="absolute -bottom-40 -left-40 w-[400px] h-[400px] rounded-full bg-sam-700/15 blur-[80px]" />
          </div>

          <div className="relative z-10 grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
            <div>
              <h2 className="text-3xl md:text-4xl font-800 tracking-tighter text-white leading-tight">
                Start automating your
                <br />
                systematic reviews today
              </h2>
              <p className="mt-4 text-zinc-400 leading-relaxed max-w-[50ch]">
                SAM is open source and designed to run on consumer hardware.
                Clone the repo, configure your model, and run your first
                meta-analysis in minutes.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 lg:justify-end">
              <a
                href="https://github.com/iqueipopg/SAM"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2.5 px-7 py-4 rounded-full bg-sam-600 text-white font-medium text-sm hover:bg-sam-500 transition-all duration-200 hover:-translate-y-[1px] active:translate-y-0 shadow-[0_4px_20px_rgba(13,148,136,0.4)]"
              >
                <GithubIcon size={16} />
                Clone repository
              </a>
              <a
                href="https://github.com/iqueipopg/SAM#readme"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2 px-7 py-4 rounded-full border border-zinc-700 text-zinc-300 font-medium text-sm hover:border-zinc-500 hover:text-white transition-all duration-200 hover:-translate-y-[1px] active:translate-y-0"
              >
                Read documentation
                <ArrowRight size={14} strokeWidth={2.5} />
              </a>
            </div>
          </div>

          {/* Code snippet */}
          <div className="relative z-10 mt-12 p-6 rounded-xl bg-zinc-900/80 border border-zinc-800 font-mono text-sm">
            <p className="text-zinc-500"># Quick start</p>
            <p className="text-sam-400 mt-1">git clone https://github.com/iqueipopg/SAM.git</p>
            <p className="text-zinc-400">cd SAM</p>
            <p className="text-zinc-400">pip install -r requirements.txt</p>
            <p className="text-zinc-400">streamlit run app.py</p>
          </div>
        </div>
      </div>
    </section>
  )
}
