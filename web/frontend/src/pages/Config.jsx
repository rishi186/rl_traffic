import { useState, useEffect } from 'react'
import { Settings, Save, RotateCcw, CheckCircle2 } from 'lucide-react'
import { fetchAPI, putAPI } from '../lib/api'

export default function Config() {
  const [config, setConfig] = useState(null)
  const [saved, setSaved] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAPI('/api/config')
      .then(setConfig)
      .finally(() => setLoading(false))
  }, [])

  const handleSave = async () => {
    try {
      await putAPI('/api/config', config)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    } catch (e) {
      alert('Failed to save: ' + e.message)
    }
  }

  const handleReset = async () => {
    try {
      const data = await fetchAPI('/api/config')
      setConfig(data)
    } catch (e) {
      alert('Failed to reload config: ' + e.message)
    }
  }

  const updateField = (section, key, value) => {
    setConfig({ ...config, [section]: { ...config[section], [key]: value } })
  }

  const updateDqn = (key, value) => {
    setConfig({
      ...config,
      training: { ...config.training, dqn: { ...config.training.dqn, [key]: value } },
    })
  }

  const updateRewardWeight = (key, value) => {
    setConfig({
      ...config,
      environment: {
        ...config.environment,
        reward_weights: { ...config.environment.reward_weights, [key]: value },
      },
    })
  }

  if (loading) return <div className="p-8 text-slate-500">Loading config...</div>
  if (!config) return <div className="p-8 text-slate-500">Failed to load config</div>

  return (
    <div className="p-8 max-w-4xl">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Settings className="w-8 h-8 text-brand-400" />
            Configuration
          </h1>
          <p className="text-slate-400 mt-1">Edit experiment configuration</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-white text-sm font-medium transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium transition-colors"
          >
            <Save className="w-4 h-4" />
            Save
          </button>
          {saved && (
            <span className="flex items-center gap-1 text-green-400 text-sm">
              <CheckCircle2 className="w-4 h-4" />
              Saved!
            </span>
          )}
        </div>
      </div>

      {/* Experiment section */}
      <ConfigSection title="Experiment">
        <ConfigField label="Name" value={config.experiment.name} onChange={(v) => updateField('experiment', 'name', v)} />
        <ConfigField label="Seed" type="number" value={config.experiment.seed} onChange={(v) => updateField('experiment', 'seed', parseInt(v))} />
        <ConfigField label="Device" value={config.experiment.device} onChange={(v) => updateField('experiment', 'device', v)} />
        <ConfigField label="Log Dir" value={config.experiment.log_dir} onChange={(v) => updateField('experiment', 'log_dir', v)} />
      </ConfigSection>

      {/* Training section */}
      <ConfigSection title="Training">
        <ConfigField label="Algorithm" value={config.training.algorithm} onChange={(v) => updateField('training', 'algorithm', v)} />
        <ConfigField label="Total Episodes" type="number" value={config.training.total_episodes} onChange={(v) => updateField('training', 'total_episodes', parseInt(v))} />
        <ConfigField label="Max Steps/Episode" type="number" value={config.training.max_steps_per_episode} onChange={(v) => updateField('training', 'max_steps_per_episode', parseInt(v))} />
        <ConfigField label="Batch Size" type="number" value={config.training.batch_size} onChange={(v) => updateField('training', 'batch_size', parseInt(v))} />
        <ConfigField label="Learning Rate" type="number" step="0.0001" value={config.training.learning_rate} onChange={(v) => updateField('training', 'learning_rate', parseFloat(v))} />
        <ConfigField label="Gamma" type="number" step="0.01" value={config.training.gamma} onChange={(v) => updateField('training', 'gamma', parseFloat(v))} />
      </ConfigSection>

      {/* DQN section */}
      {config.training.dqn && (
        <ConfigSection title="DQN Hyperparameters">
          <ConfigField label="Epsilon Start" type="number" step="0.1" value={config.training.dqn.epsilon_start} onChange={(v) => updateDqn('epsilon_start', parseFloat(v))} />
          <ConfigField label="Epsilon End" type="number" step="0.01" value={config.training.dqn.epsilon_end} onChange={(v) => updateDqn('epsilon_end', parseFloat(v))} />
          <ConfigField label="Epsilon Decay" type="number" step="0.001" value={config.training.dqn.epsilon_decay} onChange={(v) => updateDqn('epsilon_decay', parseFloat(v))} />
          <ConfigField label="Target Update Freq" type="number" value={config.training.dqn.target_update_freq} onChange={(v) => updateDqn('target_update_freq', parseInt(v))} />
          <ConfigField label="Replay Buffer Size" type="number" value={config.training.dqn.replay_buffer_size} onChange={(v) => updateDqn('replay_buffer_size', parseInt(v))} />
          <ConfigField label="Soft Update Tau" type="number" step="0.001" value={config.training.dqn.soft_update_tau} onChange={(v) => updateDqn('soft_update_tau', parseFloat(v))} />
          <ConfigField label="Grad Clip" type="number" step="1" value={config.training.dqn.grad_clip} onChange={(v) => updateDqn('grad_clip', parseFloat(v))} />
          <ConfigField label="LR Scheduler" value={config.training.dqn.lr_scheduler} onChange={(v) => updateDqn('lr_scheduler', v)} />
          <ConfigToggle label="Dueling" value={config.training.dqn.dueling} onChange={(v) => updateDqn('dueling', v)} />
          <ConfigToggle label="Double DQN" value={config.training.dqn.double_dqn} onChange={(v) => updateDqn('double_dqn', v)} />
          <ConfigToggle label="Use Attention" value={config.training.dqn.use_attention} onChange={(v) => updateDqn('use_attention', v)} />
          <ConfigToggle label="Noisy Net" value={config.training.dqn.noisy_net} onChange={(v) => updateDqn('noisy_net', v)} />
        </ConfigSection>
      )}

      {/* Reward weights */}
      {config.environment?.reward_weights && (
        <ConfigSection title="Reward Weights">
          <ConfigField label="Queue Weight" type="number" step="0.05" value={config.environment.reward_weights.queue_weight} onChange={(v) => updateRewardWeight('queue_weight', parseFloat(v))} />
          <ConfigField label="Waiting Time Weight" type="number" step="0.05" value={config.environment.reward_weights.waiting_time_weight} onChange={(v) => updateRewardWeight('waiting_time_weight', parseFloat(v))} />
          <ConfigField label="Emergency Weight" type="number" step="0.05" value={config.environment.reward_weights.emergency_weight} onChange={(v) => updateRewardWeight('emergency_weight', parseFloat(v))} />
          <ConfigField label="Throughput Bonus" type="number" step="0.01" value={config.environment.reward_weights.throughput_bonus || 0} onChange={(v) => updateRewardWeight('throughput_bonus', parseFloat(v))} />
          <ConfigField label="Switch Penalty" type="number" step="0.01" value={config.environment.reward_weights.switch_penalty || 0} onChange={(v) => updateRewardWeight('switch_penalty', parseFloat(v))} />
          <ConfigField label="Congestion Penalty" type="number" step="0.05" value={config.environment.reward_weights.congestion_penalty || 0} onChange={(v) => updateRewardWeight('congestion_penalty', parseFloat(v))} />
        </ConfigSection>
      )}

      {/* SUMO section */}
      <ConfigSection title="SUMO">
        <ConfigField label="Config File" value={config.sumo.cfg_file} onChange={(v) => updateField('sumo', 'cfg_file', v)} />
        <ConfigField label="Delta Time" type="number" value={config.sumo.delta_time} onChange={(v) => updateField('sumo', 'delta_time', parseInt(v))} />
        <ConfigField label="Min Green" type="number" value={config.sumo.min_green} onChange={(v) => updateField('sumo', 'min_green', parseInt(v))} />
        <ConfigField label="Max Green" type="number" value={config.sumo.max_green} onChange={(v) => updateField('sumo', 'max_green', parseInt(v))} />
        <ConfigField label="Density Multiplier" type="number" step="0.25" value={config.sumo.density_multiplier} onChange={(v) => updateField('sumo', 'density_multiplier', parseFloat(v))} />
      </ConfigSection>
    </div>
  )
}

function ConfigSection({ title, children }) {
  return (
    <div className="mb-8">
      <h2 className="text-lg font-semibold text-white mb-4 pb-2 border-b border-slate-800">{title}</h2>
      <div className="grid grid-cols-2 gap-4">{children}</div>
    </div>
  )
}

function ConfigField({ label, value, onChange, type = 'text', step }) {
  return (
    <div>
      <label className="block text-sm text-slate-400 mb-1.5">{label}</label>
      <input
        type={type}
        step={step}
        value={value === undefined || value === null ? '' : value}
        onChange={(e) => {
          const raw = e.target.value
          if (type === 'number') {
            if (raw === '') {
              onChange(0)
            } else {
              const parsed = step && step.includes('.') ? parseFloat(raw) : parseInt(raw)
              onChange(isNaN(parsed) ? 0 : parsed)
            }
          } else {
            onChange(raw)
          }
        }}
        className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-white text-sm focus:border-brand-500 focus:outline-none"
      />
    </div>
  )
}

function ConfigToggle({ label, value, onChange }) {
  return (
    <div className="flex items-center justify-between py-2">
      <label className="text-sm text-slate-400">{label}</label>
      <button
        onClick={() => onChange(!value)}
        className={`relative w-12 h-6 rounded-full transition-colors ${value ? 'bg-brand-600' : 'bg-slate-700'}`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${value ? 'translate-x-6' : ''}`}
        />
      </button>
    </div>
  )
}
