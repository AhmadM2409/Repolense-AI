import { useEffect, useRef, useState } from 'react'
import { apiClient } from '../services/api'
import { DurableJob } from '../types'
export function useDurableJob<T extends DurableJob>(jobId: string, load: (id: string, signal: AbortSignal) => Promise<T>, initial: T | null = null) {
  const [job, setJob] = useState<T | null>(null)
  const [error, setError] = useState('')
  const [cancelling, setCancelling] = useState(false)
  const [retrying, setRetrying] = useState(false)
  const terminal = useRef(false)
  const active = useRef<AbortController | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    active.current = controller
    terminal.current = false
    setJob(initial); setError(''); setCancelling(false); setRetrying(false)
    if (!jobId) return () => controller.abort()
    let timer: ReturnType<typeof setTimeout> | undefined
    let wake: (() => void) | undefined
    async function poll() {
      if (initial) {
        if (['complete','failed','cancelled'].includes(initial.status)) return
        await new Promise<void>(resolve => { wake = resolve; timer = setTimeout(resolve, 1000) })
      }
      while (!controller.signal.aborted && !terminal.current) {
        try {
          const result = await load(jobId, controller.signal)
          if (controller.signal.aborted || terminal.current) return
          setJob(result); setError(''); setRetrying(false)
          if (['complete','failed','cancelled'].includes(result.status)) { terminal.current = true; return }
        } catch (cause) {
          if (controller.signal.aborted) return
          const status = (cause as Error & { status?: number }).status
          setError(cause instanceof Error ? cause.message : 'Status temporarily unavailable')
          if (status && status >= 400 && status < 500 && status !== 429) { setRetrying(false); return }
          setRetrying(true)
        }
        await new Promise<void>(resolve => { wake = resolve; timer = setTimeout(resolve, 1000) })
      }
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer); wake?.() }
  }, [jobId, load, initial])
  async function cancel() {
    if (!job || cancelling) return
    const controller = active.current
    setCancelling(true)
    try {
      const result = await apiClient.cancelJob(job.id, controller?.signal)
      if (controller === active.current && !controller?.signal.aborted) {
        terminal.current = ['complete','failed','cancelled'].includes(result.status)
        setJob(current => current ? { ...current, ...result } : current)
        setError('')
      }
    } catch (cause) {
      if (controller === active.current && !controller?.signal.aborted) setError(cause instanceof Error ? cause.message : 'Cancellation failed')
    } finally { if (controller === active.current && !controller?.signal.aborted) setCancelling(false) }
  }
  return { job, error, cancel, cancelling, retrying }
}
