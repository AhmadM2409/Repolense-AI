import { useEffect, useRef, useState } from 'react'
import { apiClient } from '../services/api'
import { aiError, terminalInvestigation } from '../services/aiContracts'
import type { InvestigationStatus } from '../types'

export const POLL_INTERVAL = 2000
export function useInvestigation(repo: string, rev: string, id: string) {
  const [job, setJob] = useState<InvestigationStatus | null>(null)
  const [error, setError] = useState('')
  const [nonce, setNonce] = useState(0)
  const [cancelling, setCancelling] = useState(false)
  const mutation = useRef<AbortController | null>(null)
  const cancelLock = useRef(false)
  const stop = useRef<() => void>(() => {})
  useEffect(() => {
    const active = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    let calls = 0
    setJob(null); setError('')
    const deadline = setTimeout(() => { active.abort(); clearTimeout(timer); setError('Automatic polling paused after four minutes. Refresh status to continue.') }, 240000)
    function halt() { active.abort(); clearTimeout(timer); clearTimeout(deadline) }
    stop.current = halt
    async function poll() {
      if (active.signal.aborted) return
      if (document.hidden) { timer = setTimeout(poll, POLL_INTERVAL); return }
      if (++calls > 120) { halt(); setError('Automatic polling paused. Refresh status to continue.'); return }
      try {
        const result = await apiClient.investigation(repo, rev, id, active.signal)
        if (active.signal.aborted) return
        setJob(result)
        if (terminalInvestigation(result.status)) { clearTimeout(deadline); return }
        timer = setTimeout(poll, POLL_INTERVAL)
      } catch (cause) { if (!active.signal.aborted) { clearTimeout(deadline); setError(aiError(cause)) } }
    }
    void poll()
    return halt
  }, [repo, rev, id, nonce])
  useEffect(() => () => { mutation.current?.abort() }, [repo, rev, id])
  async function cancel() {
    if (cancelLock.current || !job || terminalInvestigation(job.status) || job.cancel_requested) return
    cancelLock.current = true; setCancelling(true); stop.current()
    const active = new AbortController(); mutation.current = active
    try {
      const result = await apiClient.cancelInvestigation(repo, rev, id, active.signal)
      if (!active.signal.aborted) { setJob(result); setError(''); if (!terminalInvestigation(result.status)) setNonce(x => x + 1) }
    } catch (cause) { if (!active.signal.aborted) setError(aiError(cause)) }
    finally { if (!active.signal.aborted) { setCancelling(false); cancelLock.current = false } }
  }
  return { job: job?.id === id ? job : null, error, cancelling, cancel, refresh: () => setNonce(x => x + 1) }
}
