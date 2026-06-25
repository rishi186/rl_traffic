import { Routes, Route, NavLink, Link } from 'react-router-dom'
import { LayoutDashboard, FlaskConical, Play, Home, Activity, Settings } from 'lucide-react'
import { cn } from './lib/utils'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Experiments from './pages/Experiments'
import Demo from './pages/Demo'
import Config from './pages/Config'

const navItems = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/experiments', label: 'Experiments', icon: FlaskConical },
  { to: '/demo', label: 'Demo', icon: Play },
  { to: '/config', label: 'Config', icon: Settings },
]

function NotFound() {
  return (
    <div className="p-8 flex items-center justify-center min-h-screen">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-slate-700 mb-4">404</h1>
        <p className="text-slate-400 mb-6">Page not found</p>
        <Link to="/" className="px-4 py-2 rounded-lg bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium">
          Back to Home
        </Link>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 h-full w-64 bg-slate-900 border-r border-slate-800 flex flex-col z-50">
        <Link to="/" className="flex items-center gap-3 p-6 border-b border-slate-800">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-white">RL Traffic</h1>
            <p className="text-xs text-slate-500">Signal Optimization</p>
          </div>
        </Link>

        <nav className="flex-1 p-4 space-y-1">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-brand-600 text-white'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                )
              }
            >
              <Icon className="w-5 h-5" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-800">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>System Online</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="ml-64 min-h-screen">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/demo" element={<Demo />} />
          <Route path="/config" element={<Config />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  )
}
