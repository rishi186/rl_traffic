import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, BarChart, Bar
} from 'recharts'
import { FlaskConical, ChevronRight, Award, Clock, TrendingDown } from 'lucide-react'
import { fetchAPI } from '../lib/api'
import { cn } from '../lib/utils'

export default function Experiments() {
  const [experiments, setExperiments] = useState([])
  const [selected, setSelected] = useState(null)
  const [detail, setDetail] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAPI('/api/experiments')
      .then((data) => {
        setExperiments(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const loadDetail = async (name) => {
    setSelected(name)
    try {
      const data = await fetchAPI(`/api/experiments/${name}`)
      setDetail(data)
    } catch (e) {
      setDetail(null)
    }
  }

  const chartData = detail?.metrics?.map((m) => ({
    episode: m.episode,
    reward: m.reward,
    waiting: m.avg_waiting_time,
    queue: m.avg_queue,
  })) || []

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-white flex items-center gap-3 mb-2">
        <FlaskConical className="w-8 h-8 text-brand-400" />
        Experiment Tracker
      </h1>
      <p className="text-slate-400 mb-8">Browse and compare past training runs</p>

      <div className="grid grid-cols-3 gap-6">
        {/* Experiment list */}
        <div className="col-span-1 space-y-2">
          {loading ? (
            <div className="text-slate-500 p-4">Loading experiments...</div>
          ) : experiments.length === 0 ? (
            <div className="text-slate-500 p-4">No experiments found. Run training first.</div>
          ) : (
            experiments.map((exp) => (
              <button
                key={exp.name}
                onClick={() => loadDetail(exp.name)}
                className={cn(
                  'w-full text-left p-4 rounded-xl border transition-colors',
                  selected === exp.name
                    ? 'bg-brand-600/10 border-brand-500/50'
                    : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                )}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-white text-sm">{exp.name}</span>
                  <ChevronRight className="w-4 h-4 text-slate-600" />
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-500">
                  {exp.has_metrics && (
                    <span className="flex items-center gap-1">
                      <Award className="w-3 h-3" />
                      {exp.num_episodes || '?'} eps
                    </span>
                  )}
                  {exp.best_reward != null && typeof exp.best_reward === 'number' && (
                    <span className="text-green-400">
                      Best: {exp.best_reward.toFixed(2)}
                    </span>
                  )}
                  {exp.has_models && (
                    <span className="text-brand-400">{exp.models?.length || 0} models</span>
                  )}
                </div>
              </button>
            ))
          )}
        </div>

        {/* Detail view */}
        <div className="col-span-2">
          {!detail ? (
            <div className="p-12 rounded-xl bg-slate-900 border border-slate-800 text-center">
              <FlaskConical className="w-12 h-12 text-slate-700 mx-auto mb-4" />
              <p className="text-slate-500">Select an experiment to view details</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Summary cards */}
              <div className="grid grid-cols-4 gap-4">
                {[
                  { label: 'Episodes', value: detail.num_episodes, icon: Clock, color: 'text-brand-400' },
                  { label: 'Best Reward', value: detail.summary.best_reward.toFixed(2), icon: Award, color: 'text-green-400' },
                  { label: 'Avg Reward', value: detail.summary.avg_reward.toFixed(2), icon: TrendingDown, color: 'text-yellow-400' },
                  { label: 'Last Reward', value: detail.summary.last_reward.toFixed(2), icon: Award, color: 'text-purple-400' },
                ].map((card) => (
                  <div key={card.label} className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                    <card.icon className={cn('w-5 h-5 mb-2', card.color)} />
                    <div className="text-2xl font-bold text-white">{card.value}</div>
                    <div className="text-xs text-slate-500">{card.label}</div>
                  </div>
                ))}
              </div>

              {/* Reward chart */}
              <div className="p-6 rounded-xl bg-slate-900 border border-slate-800">
                <h3 className="text-white font-semibold mb-4">Reward & Waiting Time</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="episode" stroke="#64748b" fontSize={12} />
                    <YAxis yAxisId="left" stroke="#64748b" fontSize={12} />
                    <YAxis yAxisId="right" orientation="right" stroke="#64748b" fontSize={12} />
                    <Tooltip
                      contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                    />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="reward" stroke="#3b82f6" strokeWidth={2} name="Reward" />
                    <Line yAxisId="right" type="monotone" dataKey="waiting" stroke="#ef4444" strokeWidth={2} name="Avg Wait Time" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Queue chart */}
              <div className="p-6 rounded-xl bg-slate-900 border border-slate-800">
                <h3 className="text-white font-semibold mb-4">Queue Length Over Episodes</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="episode" stroke="#64748b" fontSize={12} />
                    <YAxis stroke="#64748b" fontSize={12} />
                    <Tooltip
                      contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                    />
                    <Bar dataKey="queue" fill="#8b5cf6" name="Avg Queue" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
