import { useEffect } from 'react'
import { useInvestigation } from '../hooks/useInvestigation'
import { outcomeLabel, terminalInvestigation } from '../services/aiContracts'
import { GroundedAnswer } from './GroundedAnswer'

export function InvestigationRun({ repo, rev, id, onTerminal }: { repo: string; rev: string; id: string; onTerminal: () => void }) {
  const { job, error, cancelling, cancel, refresh } = useInvestigation(repo, rev, id)
  const terminal = !!job && terminalInvestigation(job.status)
  useEffect(() => { if (terminal) onTerminal() }, [terminal, onTerminal])
  const accepted = job?.status === 'complete' && job.outcome === 'completed' && job.result?.status === 'completed' && job.result.final?.status === 'answered'
  return <section aria-label="Investigation status" className="space-y-4 rounded-xl border border-slate-200 bg-slate-50 p-5">
    <div className="flex flex-wrap items-start justify-between gap-3"><div><h3 className="font-semibold">Persisted investigation</h3><p className="mt-1 break-all font-mono text-xs text-slate-500">{id}</p></div><button className="ai-secondary" onClick={refresh} disabled={cancelling}>Refresh status</button></div>
    {error ? <p role="alert" className="ai-error">{error}</p> : null}
    {!job && !error ? <p role="status">Loading saved status…</p> : null}
    {job ? <><p role="status" className="font-semibold text-slate-900">{outcomeLabel(job.outcome || job.status)}</p><p className="text-sm text-slate-500">Saved checkpoints: {job.checkpoint_count} · Model attempts: {job.inference_spent}</p>
      {!terminalInvestigation(job.status) ? <><p className="text-sm text-slate-600">Status is read from the worker every two seconds. Detailed in-flight steps are not exposed.</p><button className="ai-secondary" onClick={cancel} disabled={cancelling || job.cancel_requested}>{cancelling ? 'Requesting cancellation…' : job.cancel_requested ? 'Cancellation requested' : 'Cancel investigation'}</button><p className="text-xs text-slate-500">Cancellation is cooperative; an in-flight model request may finish before the worker stops.</p></> : null}
      {accepted && job.result?.final ? <GroundedAnswer answer={job.result.final} /> : terminalInvestigation(job.status) ? <p className="text-sm text-slate-600">No accepted answer is available for this outcome.</p> : null}
      {job.result?.steps.length ? <details className="rounded-lg border bg-white p-3"><summary className="cursor-pointer font-medium">Saved execution steps</summary><ol className="mt-3 space-y-2">{job.result.steps.map(step => <li key={step.number} className="text-sm"><span className="font-medium">Step {step.number}: {step.operation}</span>{step.origin === 'readiness' ? ' · evidence-readiness check' : ''}{step.evidence_paths.length ? <p className="break-all font-mono text-xs text-slate-500">{step.evidence_paths.join(', ')}</p> : null}</li>)}</ol></details> : null}
    </> : null}
  </section>
}
