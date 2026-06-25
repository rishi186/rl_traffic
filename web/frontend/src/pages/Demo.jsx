import { useState, useEffect } from 'react'
import { Play, Settings, Cpu, Activity, AlertCircle, Loader2 } from 'lucide-react'
import { fetchAPI, postAPI } from '../lib/api'
import { cn } from '../lib/utils'

export default function Demo() {
  const [config, setConfig] = useState(null)
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchAPI('/api/config').then(setConfig).catch(() => {})
    fetchAPI('/api/models').then((data) => {
      setModels(data)
      if (data.length > 0) setSelectedModel(data[0].path)
    }).catch(() => {})
  }, [])

  const handleRun = async () => {
    setRunning(true)
    setError(null)
    setResult(null)
    try {
      // This would trigger a simulation run — placeholder for now
      // In production, this would call /api/demo/run
      await new Promise((r) => setTimeout(r, 2000))
      setResult({
        status: 'completed',
        message: 'Simulation completed (demo mode — connect SUMO for full simulation)',
        metrics: {
          avg_waiting_time: 42.5,
          avg_queue: 8.3,
          throughput: 1250,
        },
      })
    } catch (e) {
      setError(e.message)
    }
    setRunning(false)
  }

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-white flex items-center gap-3 mb-2">
        <Play className="w-8 h-8 text-brand-400" />
        Interactive Demo
      </h1>
      <p className="text-slate-400 mb-8">Configure and run traffic simulations with trained RL agents</p>

      <div className="grid grid-cols-2 gap-6">
        {/* Configuration panel */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 space-y-6">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <Settings className="w-5 h-5 text-brand-400" />
            Simulation Settings
          </h3>

          {/* Model selection */}
          <div>
            <label className="block text-sm text-slate-400 mb-2">Trained Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg bg-slate-800 border border-slate-700 text-white text-sm focus:border-brand-500 focus:outline-none"
            >
              {models.length === 0 ? (
                <option value="">No models available</option>
              ) : (
                models.map((m) => (
                  <option key={m.path} value={m.path}>
                    {m.experiment}/{m.name}
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Simulation parameters */}
          {config && (
            <>
              <div>
                <label className="block text-sm text-slate-400 mb-2">Traffic Density</label>
                <input
                  type="range"
                  min="0.25"
                  max="2.0"
                  step="0.25"
                  defaultValue={config.sumo?.density_multiplier || 1.0}
                  className="w-full accent-brand-500"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>Light</span>
                  <span>Heavy</span>
                </div>
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-2">Max Steps</label>
                <input
                  type="number"
                  defaultValue={500}
                  className="w-full px-4 py-2.5 rounded-lg bg-slate-800 border border-slate-700 text-white text-sm focus:border-brand-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-2">Use GUI</label>
                <div className="flex gap-3">
                  <label className="flex items-center gap-2 text-sm text-slate-300">
                    <input type="radio" name="gui" defaultChecked className="accent-brand-500" />
                    Headless
                  </label>
                  <label className="flex items-center gap-2 text-sm text-slate-300">
                    <input type="radio" name="gui" className="accent-brand-500" />
                    GUI Mode
                  </label>
                </div>
              </div>
            </>
          )}

          <button
            onClick={handleRun}
            disabled={running || !selectedModel}
            className="w-full flex items-center justify-center gap-2 px-5 py-3 rounded-lg bg-brand-600 hover:bg-brand-700 text-white font-medium transition-colors disabled:opacity-50"
          >
            {running ? <Loader2 className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
            {running ? 'Running Simulation...' : 'Run Simulation'}
          </button>

          {error && (
            <div className="flex items-center gap-2 px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}
        </div>

        {/* Results panel */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800">
          <h3 className="text-white font-semibold flex items-center gap-2 mb-6">
            <Activity className="w-5 h-5 text-brand-400" />
            Results
          </h3>

          {!result && !running && (
            <div className="h-64 flex items-center justify-center text-slate-600">
              <div className="text-center">
                <Cpu className="w-12 h-12 mx-auto mb-3 text-slate-700" />
                <p>Run a simulation to see results</p>
              </div>
            </div>
          )}

          {running && (
            <div className="h-64 flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-12 h-12 mx-auto mb-3 text-brand-400 animate-spin" />
                <p className="text-slate-400">Running simulation...</p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-green-400 text-sm font-medium">
                <span className="w-2 h-2 rounded-full bg-green-500" />
                {result.status}
              </div>
              <p className="text-slate-400 text-sm">{result.message}</p>

              <div className="grid grid-cols-3 gap-4 pt-4">
                {[
                  { label: 'Avg Wait Time', value: `${result.metrics.avg_waiting_time}s`, color: 'text-red-400' },
                  { label: 'Avg Queue', value: result.metrics.avg_queue, color: 'text-yellow-400' },
                  { label: 'Throughput', value: result.metrics.throughput, color: 'text-green-400' },
                ].map((m) => (
                  <div key={m.label} className="p-4 rounded-lg bg-slate-800 border border-slate-700">
                    <div className={cn('text-2xl font-bold', m.color)}>{m.value}</div>
                    <div className="text-xs text-slate-500 mt-1">{m.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
