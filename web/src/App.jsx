import { useEffect } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Marquee from './components/Marquee'
import Pipeline from './components/Pipeline'
import Features from './components/Features'
import RunPipeline from './components/RunPipeline'
import Stats from './components/Stats'
import CTA from './components/CTA'
import Footer from './components/Footer'

gsap.registerPlugin(ScrollTrigger)

function App() {
  useEffect(() => {
    const revealElements = document.querySelectorAll('.gsap-reveal')
    revealElements.forEach((el) => {
      gsap.fromTo(el,
        { y: 60, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 1,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: el,
            start: 'top 85%',
            toggleActions: 'play none none none',
          },
        }
      )
    })

    return () => ScrollTrigger.getAll().forEach(t => t.kill())
  }, [])

  return (
    <div className="min-h-[100dvh] bg-[#FAFAFA] overflow-x-hidden">
      <Navbar />
      <Hero />
      <Marquee />
      <Pipeline />
      <Features />
      <RunPipeline />
      <Stats />
      <CTA />
      <Footer />
    </div>
  )
}

export default App
