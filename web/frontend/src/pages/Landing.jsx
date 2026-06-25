import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity, Brain, Cpu, Gauge, TrendingUp, Zap, Network,
  Layers, Target, Sparkles, ArrowRight, CheckCircle2, AlertTriangle
} from 'lucide-react'
import { fetchAPI } from '../lib/api'

const FALLBACK_PROJECT = {
  name: 'Deep Q-Network Traffic Signal Optimization',
  description: 'A production-grade DQN agent that autonomously optimises traffic signal phases across a multi-intersection urban grid using PyTorch, SUMO simulation, and the TraCI API.',
  algorithm: 'dqn',
  total_episodes: 100,
}

export default function Landing() {
  const [project, setProject] = useState(FALLBACK_PROJECT)
  const [experiments, setExperiments] = useState([])
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    Promise.all([
      fetchAPI('/api/project').then((d) => { setProject(d); setApiStatus('connected') }).catch(() => setApiStatus('offline')),
      fetchAPI('/api/experiments').then(setExperiments).catch(() => {}),
    ])
  }, [])

  const features = [
    { icon: Brain, title: 'Dueling Double DQN', desc: 'Advanced Q-learning with dueling architecture, Double DQN, and PER' },
    { icon: Layers, title: 'Attention Q-Network', desc: 'Self-attention over per-lane features for spatial reasoning' },
    { icon: Network, title: 'Multi-Agent Sharing', desc: 'Parameter sharing with agent ID embedding across intersections' },
    { icon: Target, title: 'Soft Target Updates', desc: 'Polyak averaging for stable target network updates' },
    { icon: TrendingUp, title: 'LR Scheduling', desc: 'Cosine annealing and step LR schedulers with state persistence' },
    { icon: Zap, title: 'Gradient Clipping', desc: 'Prevents exploding gradients during training' },
    { icon: Gauge, title: 'Reward Shaping', desc: 'Throughput bonus, switch penalty, congestion penalty' },
    { icon: Sparkles, title: 'Early Stopping', desc: 'Patience-based termination when reward plateaus' },
  ]

  const techStack = ['PyTorch', 'SUMO', 'TraCI', 'Gymnasium', 'FastAPI', 'React', 'TensorBoard']

  return (
    <div className="min-h-screen">
      {apiStatus === 'offline' && (
        <div className="flex items-center gap-2 px-4 py-2 bg-yellow-500/10 border-b border-yellow-500/30 text-yellow-400 text-sm justify-center">
          <AlertTriangle className="w-4 h-4" />
          Backend offline — showing demo data. Start server: <code className="px-1 py-0.5 rounded bg-yellow-500/20">python web/server.py</code>
        </div>
      )}

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-brand-950" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-brand-900/20 via-transparent to-transparent" />

        <div className="relative max-w-6xl mx-auto px-8 py-24">
          <div className="flex items-center gap-2 mb-6">
            <span className="px-3 py-1 rounded-full bg-brand-500/10 border border-brand-500/30 text-brand-400 text-xs font-medium">
              Production-Grade RL
            </span>
            <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-400 text-xs font-medium">
              PyTorch + SUMO
            </span>
          </div>

          <h1 className="text-5xl font-bold text-white mb-6 leading-tight">
            Deep Q-Network<br />
            <span className="bg-gradient-to-r from-brand-400 to-brand-600 bg-clip-text text-transparent">
              Traffic Signal Optimization
            </span>
          </h1>

          <p className="text-xl text-slate-400 max-w-2xl mb-8">
            {project?.description || 'A production-grade DQN agent that autonomously optimises traffic signal phases across a multi-intersection urban grid.'}
          </p>

          <div className="flex gap-4">
            <Link
              to="/dashboard"
              className="flex items-center gap-2 px-6 py-3 rounded-lg bg-brand-600 hover:bg-brand-700 text-white font-medium transition-colors"
            >
              <Activity className="w-5 h-5" />
              Live Dashboard
            </Link>
            <Link
              to="/experiments"
              className="flex items-center gap-2 px-6 py-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-white font-medium transition-colors"
            >
              View Experiments
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-4 gap-6 mt-16">
            {[
              { label: 'Algorithm', value: project?.algorithm?.toUpperCase() || 'DQN' },
              { label: 'Episodes', value: project?.total_episodes || '100' },
              { label: 'Experiments', value: experiments.length },
              { label: 'Features', value: '12+' },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-3xl font-bold text-white">{stat.value}</div>
                <div className="text-sm text-slate-500 mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="max-w-6xl mx-auto px-8 py-20">
        <h2 className="text-3xl font-bold text-white mb-2">Key Features</h2>
        <p className="text-slate-400 mb-12">Advanced RL techniques for traffic signal control</p>

        <div className="grid grid-cols-4 gap-6">
          {features.map(({ icon: Icon, title, desc }) => (
            <div
              key={title}
              className="p-6 rounded-xl bg-slate-900 border border-slate-800 hover:border-brand-500/50 transition-colors group"
            >
              <div className="w-12 h-12 rounded-lg bg-brand-500/10 flex items-center justify-center mb-4 group-hover:bg-brand-500/20 transition-colors">
                <Icon className="w-6 h-6 text-brand-400" />
              </div>
              <h3 className="text-white font-semibold mb-2">{title}</h3>
              <p className="text-sm text-slate-400">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Architecture */}
      <section className="max-w-6xl mx-auto px-8 py-20">
        <h2 className="text-3xl font-bold text-white mb-12">Architecture</h2>
        <div className="rounded-xl bg-slate-900 border border-slate-800 p-8">
          <div className="flex items-center justify-between gap-4">
            {[
              { icon: Cpu, label: 'SUMO Simulator', sub: 'TraCI API' },
              { icon: Gauge, label: 'Feature Pipeline', sub: 'Per-lane metrics' },
              { icon: Brain, label: 'DQN Agent', sub: 'PyTorch' },
              { icon: Activity, label: 'Target Network', sub: 'Soft update' },
            ].map((node, i) => (
              <div key={node.label} className="flex items-center gap-4 flex-1">
                <div className="flex flex-col items-center text-center flex-1">
                  <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-slate-800 to-slate-700 border border-slate-700 flex items-center justify-center mb-2">
                    <node.icon className="w-8 h-8 text-brand-400" />
                  </div>
                  <div className="text-sm font-medium text-white">{node.label}</div>
                  <div className="text-xs text-slate-500">{node.sub}</div>
                </div>
                {i < 3 && <ArrowRight className="w-5 h-5 text-slate-600" />}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="max-w-6xl mx-auto px-8 py-20">
        <h2 className="text-3xl font-bold text-white mb-8">Tech Stack</h2>
        <div className="flex flex-wrap gap-3">
          {techStack.map((tech) => (
            <span
              key={tech}
              className="px-4 py-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 text-sm font-medium"
            >
              {tech}
            </span>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800 mt-20">
        <div className="max-w-6xl mx-auto px-8 py-8 flex items-center justify-between">
          <p className="text-sm text-slate-500">RL Traffic Signal Optimization — Built with PyTorch & SUMO</p>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <CheckCircle2 className="w-4 h-4 text-green-500" />
            All systems operational
          </div>
        </div>
      </footer>
    </div>
  )
}
