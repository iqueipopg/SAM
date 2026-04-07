import GithubIcon from './GithubIcon'

export default function Footer() {
  return (
    <footer className="border-t border-zinc-100 bg-white py-12">
      <div className="max-w-[1400px] mx-auto px-6 md:px-10">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-10">
          {/* Brand */}
          <div className="md:col-span-5">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-md bg-sam-600 flex items-center justify-center">
                <span className="text-white font-bold text-xs">S</span>
              </div>
              <span className="font-semibold text-zinc-900 tracking-tight">SAM</span>
            </div>
            <p className="mt-3 text-sm text-zinc-500 leading-relaxed max-w-[40ch]">
              Systematic Agent for Meta-analysis. An open-source multi-agent
              system for automated systematic reviews in psychology and health sciences.
            </p>
            <p className="mt-4 text-xs text-zinc-400">
              Trabajo de Fin de Grado &middot; Universidad Pontificia Comillas ICAI
            </p>
          </div>

          {/* Links */}
          <div className="md:col-span-3 md:col-start-7">
            <p className="text-xs font-semibold tracking-[0.1em] uppercase text-zinc-400 mb-4">
              Resources
            </p>
            <div className="space-y-2.5">
              {[
                { label: 'GitHub', href: 'https://github.com/iqueipopg/SAM' },
                { label: 'Documentation', href: 'https://github.com/iqueipopg/SAM#readme' },
                { label: 'Issues', href: 'https://github.com/iqueipopg/SAM/issues' },
              ].map(link => (
                <a
                  key={link.label}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block text-sm text-zinc-500 hover:text-zinc-900 transition-colors duration-200"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </div>

          {/* Tech */}
          <div className="md:col-span-3">
            <p className="text-xs font-semibold tracking-[0.1em] uppercase text-zinc-400 mb-4">
              Built with
            </p>
            <div className="space-y-2.5">
              {['Python', 'R/metafor', 'HuggingFace', 'FAISS', 'Streamlit', 'LaTeX'].map(tech => (
                <span key={tech} className="block text-sm text-zinc-500">
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-12 pt-6 border-t border-zinc-100 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-zinc-400">
            &copy; 2026 Ignacio Queipo de Llano. MIT License.
          </p>
          <a
            href="https://github.com/iqueipopg/SAM"
            target="_blank"
            rel="noopener noreferrer"
            className="text-zinc-400 hover:text-zinc-600 transition-colors"
          >
            <GithubIcon size={18} />
          </a>
        </div>
      </div>
    </footer>
  )
}
