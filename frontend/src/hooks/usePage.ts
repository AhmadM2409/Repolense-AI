import { useCallback, useEffect, useRef, useState } from 'react'
import type { Page } from '../types'
import { aiError } from '../services/aiContracts'
export function usePage<T extends { id: string }>(load: (offset: number, signal: AbortSignal) => Promise<Page<T>>, enabled = true) {
  const [items, setItems] = useState<T[]>([]), [next, setNext] = useState<number | null>(null)
  const [loading, setLoading] = useState(false), [error, setError] = useState(''), [nonce, setNonce] = useState(0)
  const active = useRef<AbortController | null>(null), lock = useRef(false)
  useEffect(() => {
    const controller = new AbortController(); active.current = controller
    setItems([]); setNext(null); setError(''); setLoading(enabled); lock.current = enabled
    if (enabled) load(0, controller.signal).then(data => { if (!controller.signal.aborted) { setItems(data.items); setNext(data.next_offset) } }).catch(cause => { if (!controller.signal.aborted) setError(aiError(cause)) }).finally(() => { if (!controller.signal.aborted) { setLoading(false); lock.current = false } })
    return () => controller.abort()
  }, [load, enabled, nonce])
  async function more() {
    if (lock.current || next === null || !active.current || active.current.signal.aborted) return
    const controller = active.current; lock.current = true; setLoading(true)
    try { const data = await load(next, controller.signal); if (!controller.signal.aborted) { setItems(old => [...old, ...data.items.filter(item => !old.some(prior => prior.id === item.id))]); setNext(data.next_offset); setError('') } }
    catch (cause) { if (!controller.signal.aborted) setError(aiError(cause)) }
    finally { if (!controller.signal.aborted) { lock.current = false; setLoading(false) } }
  }
  return { items, next, loading, error, more, reload: useCallback(() => setNonce(x => x + 1), []) }
}
