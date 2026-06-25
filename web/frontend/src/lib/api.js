const API_BASE = ''

export async function fetchAPI(path) {
  try {
    const res = await fetch(`${API_BASE}${path}`)
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || `API error ${res.status}`)
    }
    return await res.json()
  } catch (e) {
    if (e instanceof TypeError && e.message.includes('fetch')) {
      console.warn(`[API] Cannot reach ${path} — backend may be offline`)
    }
    throw e
  }
}

export async function postAPI(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `API error ${res.status}`)
  }
  return await res.json()
}

export async function putAPI(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `API error ${res.status}`)
  }
  return await res.json()
}

export function connectWebSocket(onMessage, onStatus) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.hostname
  const port = window.location.port

  // In dev (Vite proxy) or production (same origin), use relative path
  // Only fall back to port 8000 if we're on a different port and no proxy
  let wsUrl
  if (port === '5173') {
    // Vite dev server — use proxy
    wsUrl = `${protocol}//${host}:5173/ws`
  } else {
    // Production or other — same origin
    wsUrl = `${protocol}//${host}:${port || '8000'}/ws`
  }

  let ws
  let reconnectTimer = null
  let closed = false

  function connect() {
    if (closed) return
    try {
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('[WS] Connected')
        onStatus?.('connected')
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessage(data)
        } catch (e) {
          console.warn('[WS] Failed to parse message:', e)
        }
      }

      ws.onerror = () => {
        onStatus?.('error')
      }

      ws.onclose = () => {
        onStatus?.('disconnected')
        if (!closed) {
          if (reconnectTimer) clearTimeout(reconnectTimer)
          reconnectTimer = setTimeout(connect, 3000)
        }
      }
    } catch (e) {
      console.warn('[WS] Failed to connect:', e)
      onStatus?.('error')
      if (!closed) {
        reconnectTimer = setTimeout(connect, 3000)
      }
    }
  }

  connect()

  return {
    close: () => {
      closed = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      if (ws) ws.close()
    },
  }
}
