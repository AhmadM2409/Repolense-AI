import { useEffect, useRef, useState } from 'react'
import { apiClient } from '../services/api'
import { aiError } from '../services/aiContracts'
import { GroundedAnswer } from './GroundedAnswer'
import type { QuestionPolicy, RepositoryAnswer } from '../types'

export function RepositoryQuestions({ repositoryId, revisionId }: { repositoryId: string; revisionId: string }) {
  const operation = useRef<AbortController | null>(null)
  const submitting = useRef(false)
  const [policy, setPolicy] = useState<QuestionPolicy | null>(null)
  const [provider, setProvider] = useState<'lmstudio' | 'gemini'>('gemini')
  const [question, setQuestion] = useState('')
  const [consent, setConsent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [answer, setAnswer] = useState<RepositoryAnswer | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    operation.current?.abort(); submitting.current = false
    setPolicy(null); setConsent(false); setAnswer(null); setError(''); setQuestion(''); setBusy(false)
    apiClient.questionPolicy(repositoryId, revisionId, controller.signal).then(data => {
      if (!controller.signal.aborted) { setPolicy(data); setProvider(data.default_provider) }
    }).catch(cause => { if (!controller.signal.aborted) setError(aiError(cause)) })
    return () => { controller.abort(); operation.current?.abort() }
  }, [repositoryId, revisionId])
  async function ask(event: React.FormEvent) {
    event.preventDefault()
    if (submitting.current || !policy || !question.trim() || (provider === 'gemini' && (!policy.hosted_available || !consent))) return
    submitting.current = true
    const active = new AbortController(); operation.current = active
    setBusy(true); setError(''); setAnswer(null)
    try {
      const result = await apiClient.askRepository(repositoryId, revisionId, {
        question: question.trim(), retrieval_mode: 'keyword', generation_provider: provider,
        ...(provider === 'gemini' ? { hosted_processing_consent: 'gemini-source-processing-v1' as const } : {})
      }, active.signal)
      if (!active.signal.aborted) setAnswer(result)
    } catch (cause) { if (!active.signal.aborted) setError(aiError(cause)) }
    finally { if (operation.current === active) { submitting.current = false; if (!active.signal.aborted) { setBusy(false); setConsent(false) } } }
  }
  return <section aria-label="Repository questions" className="space-y-5">
    <div><h2 className="text-xl font-semibold text-slate-900">Ask about this revision</h2><p className="mt-1 text-sm text-slate-500">A focused answer grounded in the selected commit. Every accepted answer includes source evidence.</p></div>
    <form onSubmit={ask} className="space-y-4">
      <label className="block text-sm font-medium text-slate-700">Provider<select className="ai-input mt-1 max-w-xs" disabled={busy || !policy} value={provider} onChange={e => { setProvider(e.target.value as 'lmstudio' | 'gemini'); setConsent(false); setAnswer(null) }}><option value="lmstudio">Local Qwen</option><option value="gemini" disabled={!policy?.hosted_available}>Google Gemini (hosted){policy && !policy.hosted_available ? ' — unavailable' : ''}</option></select></label>
      <label className="block text-sm font-medium text-slate-700">Question<textarea className="ai-input mt-1 min-h-28" maxLength={1024} required disabled={busy} value={question} placeholder="What does this function do, and where is its configuration defined?" onChange={e => { setQuestion(e.target.value); setConsent(false) }} /></label>
      {provider === 'gemini' ? <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950"><p>{policy?.notice || 'Google receives this question and selected source excerpts. Availability is being checked.'}</p><label className="mt-3 flex items-start gap-2"><input className="mt-1" type="checkbox" checked={consent} disabled={busy || !policy?.hosted_available} onChange={e => setConsent(e.target.checked)} /><span>I authorize sending this question and selected source excerpts for this revision to Google.</span></label></div> : null}
      <button className="ai-primary" type="submit" disabled={!policy || busy || !question.trim() || (provider === 'gemini' && (!consent || !policy.hosted_available))}>{busy ? 'Checking evidence…' : 'Ask'}</button>
      {busy ? <p role="status" className="text-sm text-slate-600">Retrieving sources and validating the answer. This request will not be retried automatically.</p> : null}
    </form>
    {error ? <p role="alert" className="ai-error">{error}</p> : null}
    {answer ? <GroundedAnswer answer={answer} /> : null}
  </section>
}
