import { useEffect, useRef, useState } from 'react'
import { apiClient } from '../services/api'
import { aiError } from '../services/aiContracts'
import type { QuestionPolicy } from '../types'
export function InvestigationComposer({ repo, rev, onStarted, refreshHistory }: { repo: string; rev: string; onStarted: (id: string) => void; refreshHistory: () => void }) {
  const [policy, setPolicy] = useState<QuestionPolicy | null>(null), [objective, setObjective] = useState('')
  const [consent, setConsent] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState('')
  const active = useRef<AbortController | null>(null), lock = useRef(false)
  useEffect(() => { const controller = new AbortController(); apiClient.questionPolicy(repo, rev, controller.signal).then(data => { if (!controller.signal.aborted) setPolicy(data) }).catch(cause => { if (!controller.signal.aborted) setError(aiError(cause)) }); return () => { controller.abort(); active.current?.abort() } }, [repo, rev])
  const available = policy?.hosted_available && policy.durable_investigations_available
  async function submit(event: React.FormEvent) {
    event.preventDefault(); if (lock.current || !available || !consent || !objective.trim()) return
    lock.current = true; setBusy(true); setError(''); const controller = new AbortController(); active.current = controller
    try { const result = await apiClient.startInvestigation(repo, rev, objective.trim(), controller.signal); if (!controller.signal.aborted) { onStarted(result.id); refreshHistory() } }
    catch (cause) { if (!controller.signal.aborted) { setError(aiError(cause)); refreshHistory() } }
    finally { if (!controller.signal.aborted) { lock.current = false; setBusy(false); setConsent(false) } }
  }
  return <section aria-label="Start investigation" className="space-y-4"><div><h2 className="text-xl font-semibold text-slate-900">Investigate across sources</h2><p className="mt-1 text-sm text-slate-500">A bounded multi-step investigation, saved on the server and processed by an independent worker.</p></div>
    {policy && !available ? <p role="status" className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-950">Durable investigations are not enabled for this backend. Saved investigations remain available below.</p> : null}
    <form onSubmit={submit} className="space-y-4"><label className="block text-sm font-medium text-slate-700">Investigation objective<textarea className="ai-input mt-1 min-h-28" value={objective} required maxLength={1024} disabled={busy} placeholder="Trace this function to its configuration and explain the decision it makes." onChange={e => { setObjective(e.target.value); setConsent(false) }} /></label>
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950"><p>{policy?.notice || 'Availability and hosted-processing policy are being checked.'}</p><label className="mt-3 flex items-start gap-2"><input className="mt-1" type="checkbox" disabled={busy || !available} checked={consent} onChange={e => setConsent(e.target.checked)} /><span>I authorize Google to process this objective and selected source excerpts for this investigation and revision.</span></label></div>
      <button className="ai-primary" type="submit" disabled={!available || busy || !objective.trim() || !consent}>{busy ? 'Saving investigation…' : 'Start investigation'}</button><p className="text-xs text-slate-500">No automatic submission retries. If submission cannot be confirmed, check saved history before creating another investigation.</p>
    </form>{error ? <p role="alert" className="ai-error">{error}</p> : null}</section>
}
