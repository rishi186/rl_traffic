import { useState, useEffect } from 'react'
import { Play, Pause, AlertCircle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'
import { fetchAPI } from '../lib/api'

const number = (value) => value == null ? 'N/A' : Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })
const colors = { G: '#22c55e', g: '#86efac', y: '#facc15', Y: '#facc15', r: '#ef4444' }

export default function Demo() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  useEffect(() => { fetchAPI('/api/benchmark/replay').then(setData).catch(e => setError(e.message)) }, [])
  useEffect(() => {
    if (!playing || !data) return
    const timer = setInterval(() => setIndex(i => {
      if (i >= data.panels[0].timeline.length - 1) { setPlaying(false); return i }
      return i + 1
    }), 120)
    return () => clearInterval(timer)
  }, [playing, data])
  if (error) return <div className="p-8 text-amber-300" role="alert"><AlertCircle className="mb-3" />{error}</div>
  if (!data) return <p className="p-8 text-slate-400">Loading verified paired replay…</p>
  const ticks = data.panels[0].timeline
  const tick = ticks[index]
  const chart = ticks.map((t, i) => ({ second: t.elapsed_s, fixed: t.network_stopped,
    dqn: data.panels[1].timeline[i].network_stopped,
    fixedQueue: t.signal.queue, dqnQueue: data.panels[1].timeline[i].signal.queue }))
  return <div className="p-5 md:p-8 space-y-6 text-slate-200">
    <div><p className="text-brand-400 text-sm uppercase tracking-widest">Recorded SUMO benchmark</p>
      <h1 className="text-3xl font-bold text-white mt-2">Two policies. One clock.</h1>
      <p className="text-slate-400 mt-2">{data.label} · {data.scenario} demand · matched seed {data.seed} · simulated time {tick.simulated_second}s</p>
      <p className="text-sm text-slate-500 mt-1">This replays measured artifacts. Final trip statistics below cover the whole episode.</p></div>
    <div className="flex gap-4 items-center bg-slate-900 p-4 rounded-xl border border-slate-800">
      <button aria-label={playing ? 'Pause replay' : 'Play replay'} className="p-3 bg-brand-600 rounded-lg" onClick={() => {
        if (index === ticks.length - 1) setIndex(0)
        setPlaying(!playing)
      }}>{playing ? <Pause size={18} /> : <Play size={18} />}</button>
      <input aria-label="Replay simulated second" type="range" min="0" max={ticks.length - 1} value={index}
        className="flex-1" onChange={e => { setPlaying(false); setIndex(Number(e.target.value)) }} />
      <span className="tabular-nums">{tick.elapsed_s} / {ticks.length}s</span>
    </div>
    <div className="grid md:grid-cols-2 gap-5">{data.panels.map((panel, p) => {
      const t = panel.timeline[index], m = panel.metrics
      const action = panel.actions.find(a => a.elapsed_s >= t.elapsed_s)
      return <section key={panel.policy} className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
        <div className="flex justify-between"><h2 className="text-xl font-semibold uppercase">{panel.policy === 'fixed' ? 'Fixed time' : `${panel.policy} · train seed ${m.training_seed}`}</h2>
          <span className="text-sm text-slate-400">t = {t.simulated_second}s</span></div>
        <p className="text-xs text-slate-400 break-all">Signal {data.signal_id} · SUMO link order</p>
        <div className="flex flex-wrap gap-1.5" aria-label={`Signal state ${t.signal.state}`}>
          {[...t.signal.state].map((state, i) => <span key={i} title={`Link ${i}: ${state}`}
            className="w-6 h-6 rounded-full text-black text-xs grid place-content-center" style={{ background: colors[state] || '#94a3b8' }}>{state}</span>)}
        </div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>Network stopped now<strong className="block text-2xl text-white">{t.network_stopped}</strong></div>
          <div>Selected signal queue now<strong className="block text-2xl text-white">{t.signal.queue}</strong></div>
        </div>
        <p className="text-xs text-slate-400">{p === 0 ? 'SUMO runs the common fixed program.' : `Decision window request / switch initiated in window: ${action?.requested[data.signal_id] ?? '—'} / ${action?.applied[data.signal_id] ?? '—'} (0 hold, 1 switch).`}</p>
        <dl className="text-sm grid grid-cols-2 gap-y-2 border-t border-slate-800 pt-4">
          <dt>Final stopped vehicle-seconds</dt><dd>{number(m.network_stopped_vehicle_seconds)}</dd>
          <dt>Final completed / scheduled</dt><dd>{m.completed} / {m.scheduled_demand}</dd>
          <dt>Final unfinished / uninserted</dt><dd>{m.unfinished} / {m.uninserted}</dd>
          <dt>Completed trips: mean wait</dt><dd>{number(m.trip_wait_completed_mean_s)} s</dd>
          <dt>Completed trips: p95 wait</dt><dd>{number(m.trip_wait_completed_p95_s)} s</dd>
          <dt>Completed trips: mean duration</dt><dd>{number(m.trip_duration_completed_mean_s)} s</dd>
        </dl>
      </section>
    })}</div>
    <section className="p-5 bg-slate-900 rounded-xl border border-slate-800">
      <h2 className="font-semibold mb-5">Network stopped vehicles and selected signal queues</h2>
      <div style={{ width: '100%', height: 280 }}><ResponsiveContainer><LineChart data={chart}>
        <XAxis dataKey="second" tick={{ fill: '#94a3b8' }} /><YAxis tick={{ fill: '#94a3b8' }} />
        <Tooltip contentStyle={{ background: '#0f172a', borderColor: '#334155' }} /><Legend />
        <Line dataKey="fixed" name="Fixed: network stopped" stroke="#fbbf24" dot={false} isAnimationActive={false} />
        <Line dataKey="dqn" name="DQN: network stopped" stroke="#38bdf8" dot={false} isAnimationActive={false} />
        <Line dataKey="fixedQueue" name="Fixed: signal queue" stroke="#fbbf24" strokeDasharray="4 4" dot={false} isAnimationActive={false} />
        <Line dataKey="dqnQueue" name="DQN: signal queue" stroke="#38bdf8" strokeDasharray="4 4" dot={false} isAnimationActive={false} />
        <ReferenceLine x={tick.elapsed_s} stroke="#fff" />
      </LineChart></ResponsiveContainer></div>
    </section>
    <details className="text-sm text-slate-400 bg-slate-900 rounded-xl p-5" open>
      <summary className="cursor-pointer text-slate-200">Provenance and limits</summary>
      <p className="mt-3">{data.provenance.sumo} · Python {data.provenance.python.split(' ')[0]} · Git {data.provenance.git_revision.slice(0, 12)}</p>
      <p className="break-all mt-2">Match: {data.match_id}<br />Raw CSV SHA-256: {data.raw_csv_sha256}<br />Source digest: {data.provenance.source_digest}</p>
      <div className="mt-3 space-y-2">{Object.entries(data.models || {}).map(([path, hash]) => <p className="break-all" key={path}>Checkpoint SHA-256: {hash}</p>)}</div>
      <ul className="list-disc pl-5 mt-3 space-y-1">{data.limitations.map(l => <li key={l}>{l}</li>)}</ul>
    </details>
  </div>
}
