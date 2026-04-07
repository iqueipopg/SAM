import { useState, useEffect } from 'react'
import { Menu, X } from 'lucide-react'
import GithubIcon from './GithubIcon'

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const links = [
    { label: 'Pipeline', href: '#pipeline' },
    { label: 'Features', href: '#features' },
    { label: 'Run', href: '#run' },
    { label: 'GitHub', href: 'https://github.com/iqueipopg/SAM', external: true },
  ]

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-white/80 backdrop-blur-xl border-b border-zinc-200/50 shadow-[0_1px_3px_rgba(0,0,0,0.04)]'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-[1400px] mx-auto px-6 md:px-10">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <a href="#" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-sam-600 flex items-center justify-center transition-transform duration-200 group-hover:scale-[0.96] group-active:scale-[0.92]">
              <span className="text-white font-bold text-sm tracking-tight">S</span>
            </div>
            <span className="font-semibold text-zinc-900 tracking-tight">SAM</span>
          </a>

          {/* Desktop links */}
          <div className="hidden md:flex items-center gap-8">
            {links.map(link => (
              <a
                key={link.label}
                href={link.href}
                target={link.external ? '_blank' : undefined}
                rel={link.external ? 'noopener noreferrer' : undefined}
                className="text-sm text-zinc-500 hover:text-zinc-900 transition-colors duration-200"
              >
                {link.label}
              </a>
            ))}
            <a
              href="https://github.com/iqueipopg/SAM"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-800 transition-all duration-200 hover:-translate-y-[1px] active:translate-y-0"
            >
              <GithubIcon size={15} />
              Star on GitHub
            </a>
          </div>

          {/* Mobile toggle */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 text-zinc-600 hover:text-zinc-900"
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        {/* Mobile menu */}
        {mobileOpen && (
          <div className="md:hidden pb-6 pt-2 border-t border-zinc-100">
            <div className="flex flex-col gap-4">
              {links.map(link => (
                <a
                  key={link.label}
                  href={link.href}
                  onClick={() => setMobileOpen(false)}
                  className="text-sm text-zinc-600 hover:text-zinc-900 transition-colors"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}
