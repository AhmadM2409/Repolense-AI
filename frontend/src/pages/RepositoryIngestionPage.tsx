import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { apiClient } from '../services/api'
import { IngestionJob, IngestedSourceDetail } from '../types'
import { useDurableJob } from '../hooks/useDurableJob'
import { JobProgress } from '../components/JobProgress'
import { RepositoryQuestions } from '../components/RepositoryQuestions'
const loadIngestion = (id: string, signal: AbortSignal) => apiClient.getIngestion(id, signal)

export function RepositoryIngestionPage() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const [analyzing, setAnalyzing] = useState(false)
  const [repository, setRepository] = useState('')
  const [profileId, setProfileId] = useState(params.get('profile_id') || '')
  const [jobId, setJobId] = useState(params.get('job_id') || '')
  const [submitted, setSubmitted] = useState<IngestionJob | null>(null)
  const { job, error: pollingError, cancel, cancelling, retrying } = useDurableJob(jobId, loadIngestion, submitted)
  const [detail, setDetail] = useState<IngestedSourceDetail | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)

  useEffect(() => () => controller.current?.abort(), [])

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    controller.current?.abort()
    const active = new AbortController()
    controller.current = active
    setBusy(true); setError(''); setSubmitted(null); setJobId(''); setDetail(null)
    try {
      let ownerProfile = profileId.trim()
      if (!ownerProfile) {
        const profile = await apiClient.createProfile(new FormData())
        ownerProfile = profile.id
        if (active.signal.aborted) return
        setProfileId(ownerProfile)
      }
      const result = await apiClient.submitRepository(ownerProfile, repository.trim(), active.signal)
      if (active.signal.aborted) return
      setSubmitted(result)
      setJobId(result.id)
      setParams({ profile_id: ownerProfile, job_id: result.id }, { replace: true })
    } catch (cause) {
      if (!active.signal.aborted) setError(cause instanceof Error ? cause.message : 'Ingestion request failed')
    } finally {
      if (!active.signal.aborted) setBusy(false)
    }
  }

  async function startAnalysis() {
    if (!job) return
    setAnalyzing(true); setError('')
    try {
      const analysis = await apiClient.startAnalysis(job.id, controller.current?.signal)
      if (!controller.current?.signal.aborted) navigate('/analyses/' + analysis.id)
    } catch (cause) {
      if (!controller.current?.signal.aborted) setError(cause instanceof Error ? cause.message : 'Analysis submission failed')
    } finally { if (!controller.current?.signal.aborted) setAnalyzing(false) }
  }

  async function preview(sourceId: string) {
    if (!job) return
    const active = controller.current
    try {
      const result = await apiClient.getIngestedSource(job.id, sourceId, active?.signal)
      if (active && !active.signal.aborted) setDetail(result)
    } catch (cause) {
      if (active && !active.signal.aborted) setError(cause instanceof Error ? cause.message : 'Source could not be loaded')
    }
  }

  return (
    <main className="max-w-3xl mx-auto p-8 space-y-6">
      <Link to="/dashboard" className="text-blue-600">Back to dashboard</Link>
      <h1 className="text-2xl font-bold">Ingest a GitHub repository</h1>
      <p>Import supported public repository source files at a verified commit.</p>
      <form onSubmit={submit} className="space-y-4">
        <div>
          <label htmlFor="repository" className="block font-medium">Public GitHub repository</label>
          <input id="repository" required maxLength={300} value={repository} disabled={busy}
            onChange={event => setRepository(event.target.value)}
            placeholder="owner/repo or https://github.com/owner/repo"
            className="w-full border rounded p-2" />
        </div>
        <div>
          <label htmlFor="profile" className="block font-medium">Profile ID (optional)</label>
          <input id="profile" value={profileId} disabled={busy}
            onChange={event => setProfileId(event.target.value)} className="w-full border rounded p-2" />
          <p className="text-sm text-gray-600">Leave blank to create a profile. Its ID is retained here for repeat submissions.</p>
        </div>
        <button disabled={busy || (!!job && ['pending','processing'].includes(job.status))} className="bg-blue-600 text-white rounded px-4 py-2">
          {busy ? 'Ingesting…' : 'Ingest repository'}
        </button>
      </form>
      {busy ? <p role="status" aria-live="polite">Fetching and verifying repository content…</p> : null}
      {error ? <p role="alert" className="text-red-700">{error}</p> : null}
      {job ? <section className="space-y-2">
        <h2 className="text-xl font-semibold">Ingestion status: {job.status}</h2>
        <p className="break-all">Job ID: {job.id}</p>
        <JobProgress job={job} cancel={cancel} cancelling={cancelling} />
        {job.status === 'failed' ? <p role="alert" className="text-red-700">{job.error_message || 'Ingestion failed'}</p> : null}
        {job.status === 'complete' && job.ingestion_result?.files ? <>
          <button disabled={analyzing} onClick={startAnalysis} className="bg-blue-600 text-white px-4 py-2 rounded">{analyzing ? 'Starting analysis…' : 'Analyze this revision'}</button>
          {job.ingestion_result.repository_id && job.revision_id ? <RepositoryQuestions key={job.ingestion_result.repository_id + '/' + job.revision_id} repositoryId={job.ingestion_result.repository_id} revisionId={job.revision_id} /> : null}
          <p>Commit: <code className="break-all">{job.ingestion_result.commit_sha}</code></p>
          <p>{job.ingestion_result.file_count} files, {job.ingestion_result.chunk_count} chunks.
            {job.ingestion_result.deduplicated ? ' Existing verified revision reused.' : ''}</p>
          <p>{job.ingestion_result.skipped?.length || 0} entries excluded by file, encoding or depth limits.</p>
          <ul>{job.ingestion_result.files.map(file => <li key={file.id}>
            <button className="text-blue-600 underline" onClick={() => preview(file.id)}>{file.path}</button>
          </li>)}</ul>
        </> : null}
      </section> : null}
      {detail ? <section>
        <h2 className="font-semibold">{detail.source.path}</h2>
        <p>SHA-256: <code className="break-all">{detail.source.content_hash}</code></p>
        <pre className="overflow-auto border p-3 text-sm">{detail.source.text}</pre>
      </section> : null}
    </main>
  )
}
