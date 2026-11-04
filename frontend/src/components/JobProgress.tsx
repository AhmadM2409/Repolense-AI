import { DurableJob } from '../types'
export function JobProgress({ job, cancel, cancelling }: { job: DurableJob; cancel: () => void; cancelling: boolean }) {
  const active = ['pending','processing'].includes(job.status)
  return <section aria-label="Background job">
    <p role="status">{job.status === 'pending'
      ? job.retry_at ? 'Retry waiting' : 'Queued — waiting for a worker'
      : job.status === 'cancelled' ? 'Cancelled' : job.status}</p>
    <p>Stage: {job.progress_stage || 'queued'}</p>
    {typeof job.progress_pct === 'number' ? <progress max={100} value={job.progress_pct} aria-label="Job progress" /> : null}
    <p>Attempts: {job.attempt_count ?? 0} / {job.max_attempts ?? 3}</p>
    {job.retry_at ? <p>Next retry: {job.retry_at}</p> : null}
    {job.error_message ? <p>{job.error_message}</p> : null}
    {job.cancel_requested_at && active ? <p>Cancellation requested; waiting for a safe checkpoint.</p> : null}
    {active ? <button className="border rounded px-3 py-2" onClick={cancel} disabled={cancelling || !!job.cancel_requested_at}>
      {cancelling ? 'Requesting cancellation…' : 'Cancel job'}
    </button> : null}
  </section>
}
