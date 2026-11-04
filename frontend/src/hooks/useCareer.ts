import { useCallback, useEffect, useRef, useState } from 'react'
import { careerError } from '../services/careerContracts'
export function useCareerAction() {
  const [busy,setBusy] = useState(false), [error,setError] = useState('')
  const active = useRef<AbortController | null>(null), lock = useRef(false)
  useEffect(() => () => active.current?.abort(),[])
  async function run<T>(task: (signal: AbortSignal) => Promise<T>, accept: (value:T) => void) {
    if (lock.current) return
    lock.current = true; const controller = new AbortController(); active.current = controller; setBusy(true); setError('')
    try { const value = await task(controller.signal); if (!controller.signal.aborted) accept(value) }
    catch (cause) { if (!controller.signal.aborted) setError(careerError(cause)) }
    finally { if (!controller.signal.aborted) { lock.current = false; setBusy(false) } }
  }
  return {busy,error,run}
}
export function useCareerRead<T>(load: (signal:AbortSignal) => Promise<T>, poll?: (value:T, forced:boolean) => boolean, forced = false) {
  const [value,setValue] = useState<T | null>(null), [error,setError] = useState(''), [loading,setLoading] = useState(true), [paused,setPaused] = useState(false), [nonce,setNonce] = useState(0)
  useEffect(() => {
    const controller = new AbortController(); let timer: ReturnType<typeof setTimeout> | undefined, count = 0
    setLoading(true); setError(''); setPaused(false)
    async function read() {
      try {
        const data = await load(controller.signal)
        if (controller.signal.aborted) return
        setValue(data); setLoading(false); count++
        if (poll?.(data,forced)) {
          if (count < 40) timer = setTimeout(read,3000)
          else setPaused(true)
        }
      } catch (cause) { if (!controller.signal.aborted) { setError(careerError(cause)); setValue(null); setLoading(false) } }
    }
    void read()
    return () => { controller.abort(); clearTimeout(timer) }
  },[load,poll,forced,nonce])
  return {value,error,loading,paused,refresh:useCallback(() => setNonce(n=>n+1),[])}
}
