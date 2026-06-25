import { useState, useEffect, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, Legend
} from 'recharts'
import { Activity, Play, Square, TrendingUp, AlertCircle, Loader2, WifiOff, Wifi } from 'lucide-react'
import { fetchAPI, postAPI, connectWebSocket } from '../lib/api'
import { cn } from '../lib/utils'

export default function Dashboard() {
  const [liveState, setLiveState] = useState({
    is_training: false,
    current_episode: 0,
    total_episodes: 0,
    metrics_history: [],
    latest_metrics: {},
    training_log: [],
  })
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState(null)
  const [wsStatus, setWsStatus] = useState('connecting')
  const logRef = useRef(null)

  // WebSocket for live updates
  useEffect(() => {
    const ws = connectWebSocket(
      (data) => {
        if (data.type === 'state') {
          setLiveState(data.data)
        }
      },
      (status) => setWsStatus(status)
    )
    return () => ws.close()
  }, [])

  // Polling fallback when WS is disconnected
  useEffect(() => {
    if (wsStatus === 'connected') return
    const interval = setInterval(() => {
      fetchAPI('/api/live-state').then(setLiveState).catch(() => {})
    }, 5000)
    return () => clearInterval(interval)
  }, [wsStatus])

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [liveState.training_log])

  const handleStart = async () => {
    setStarting(true)
    setError(null)
    try {
      await postAPI('/api/train', { config_path: 'config.yaml' })
    } catch (e) {
      setError(e.message)
    }
    setStarting(false)
  }

  const handleStop = async () => {
    try {
      await postAPI('/api/train/stop', {})
    } catch (e) {
      setError(e.message)
    }
  }

  const chartData = liveState.metrics_history.map((m) => ({
    episode: m.episode,
    reward: m.reward,
  }))

  const isTraining = liveState.is_training

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Activity className="w-8 h-8 text-brand-400" />
            Training Dashboard
          </h1>
          <p className="text-slate-400 mt-1">Real-time training metrics and monitoring</p>
        </div>

        <div className="flex gap-3">
          {error && (
            <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}
          {!isTraining ? (
            <button
              onClick={handleStart}
              disabled={starting}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-green-600 hover:bg-green-700 text-white font-medium transition-colors disabled:opacity-50"
            >
              {starting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Play className="w-5 h-5" />}
              Start Training
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-red-600 hover:bg-red-700 text-white font-medium transition-colors"
            >
              <Square className="w-5 h-5" />
              Stop Training
            </button>
          )}
        </div>
      </div>

      {/* Status bar */}
      <div className={cn(
        'flex items-center gap-3 px-4 py-3 rounded-lg mb-6 border',
        isTraining
          ? 'bg-green-500/10 border-green-500/30 text-green-400'
          : 'bg-slate-900 border-slate-800 text-slate-400'
      )}>
        <div className={cn(
          'w-3 h-3 rounded-full',
          isTraining ? 'bg-green-500 animate-pulse' : 'bg-slate-600'
        )} />
        <span className="font-medium">
          {isTraining ? `Training in progress — Episode ${liveState.current_episode}` : 'Idle'}
        </span>
        <span className="ml-auto flex items-center gap-1.5 text-xs">
          {wsStatus === 'connected' ? (
            <><Wifi className="w-3.5 h-3.5 text-green-400" /> <span className="text-green-400">Live</span></>
          ) : (
            <><WifiOff className="w-3.5 h-3.5 text-yellow-400" /> <span className="text-yellow-400">Polling</span></>
          )}
        </span>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {[
          { label: 'Episode', value: liveState.current_episode, color: 'text-brand-400' },
          { label: 'Latest Reward', value: liveState.latest_metrics?.reward?.toFixed(2) || '—', color: 'text-green-400' },
          { label: 'Episodes Logged', value: liveState.metrics_history.length, color: 'text-yellow-400' },
          { label: 'Log Lines', value: liveState.training_log.length, color: 'text-purple-400' },
        ].map((card) => (
          <div key={card.label} className="p-5 rounded-xl bg-slate-900 border border-slate-800">
            <div className="text-sm text-slate-500 mb-1">{card.label}</div>
            <div className={cn('text-2xl font-bold', card.color)}>{card.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Reward chart */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800">
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-brand-400" />
            Reward Curve
          </h3>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="episode" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                />
                <Area
                  type="monotone"
                  dataKey="reward"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  fill="url(#rewardGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-slate-600">
              No training data yet. Start training to see live metrics.
            </div>
          )}
        </div>

        {/* Training log */}
        <div className="p-6 rounded-xl bg-slate-900 border border-slate-800">
          <h3 className="text-white font-semibold mb-4">Training Log</h3>
          <div
            ref={logRef}
            className="h-[300px] overflow-y-auto bg-slate-950 rounded-lg p-4 font-mono text-xs text-slate-400 space-y-1"
          >
            {liveState.training_log.length > 0 ? (
              liveState.training_log.slice(-100).map((line, i) => (
                <div key={i} className="leading-relaxed">{line}</div>
              ))
            ) : (
              <div className="text-slate-600">No logs yet. Start training to see output.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
