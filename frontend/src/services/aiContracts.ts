import type { InvestigationStatus, RepositoryAnswer } from '../types'

export function aiError(cause: unknown): string {
  const status = (cause as { status?: number })?.status
  if (status === 401) return 'Your session expired. Sign in again to continue.'
  if (status === 403 || status === 404) return 'This repository, revision or investigation is unavailable to your account.'
  if (status === 409) return 'This operation is unavailable or its repository scope changed. Refresh the context before continuing.'
  if (status === 422 || status === 413) return 'The request is invalid or exceeds the allowed limits.'
  if (status === 502) return 'The response could not be validated and has been withheld.'
  if (status === 503) return 'The model provider is unavailable. No answer was accepted.'
  if (status === 429) return 'The service is busy. Wait before trying again.'
  return 'The request could not be completed. Check the connection and try again explicitly.'
}

function invalid(): never { throw Object.assign(new Error('Invalid scoped response'), { status: 502 }) }
export function validateAnswer(data: RepositoryAnswer, repo: string, rev: string): RepositoryAnswer {
  const raw = data as RepositoryAnswer & { repository_id: string; commit_sha: string }
  if (!raw || raw.repository_id !== repo || raw.revision_id !== rev || !/^[a-f0-9]{40}$/.test(raw.commit_sha) ||
      !['answered', 'insufficient_evidence', 'validation_failure', 'provider_failure'].includes(raw.status) ||
      typeof raw.answer !== 'string' || typeof raw.uncertainty !== 'string' || !Array.isArray(raw.citations) || raw.citations.length > 20) invalid()
  if (raw.status === 'answered' && (!raw.answer.trim() || !raw.citations.length)) invalid()
  for (const c of raw.citations) {
    if (!c || c.repository_id !== repo || c.revision_id !== rev || c.commit_sha !== raw.commit_sha ||
        typeof c.path !== 'string' || typeof c.quote !== 'string' || typeof c.chunk_id !== 'string' || typeof c.source_id !== 'string' ||
        !Number.isInteger(c.line_start) || !Number.isInteger(c.line_end) || c.line_start < 1 || c.line_end < c.line_start) invalid()
  }
  return data
}
export function validateInvestigation(data: InvestigationStatus, repo: string, rev: string, id: string): InvestigationStatus {
  if (!data || data.id !== id || data.repository_id !== repo || data.revision_id !== rev ||
      !['pending', 'processing', 'complete', 'failed', 'cancelled'].includes(data.status) ||
      typeof data.cancel_requested !== 'boolean' || typeof data.deadline_at !== 'string' || !Number.isFinite(Date.parse(data.deadline_at)) ||
      (data.outcome !== null && typeof data.outcome !== 'string') ||
      !Number.isInteger(data.inference_spent) || data.inference_spent < 0 || data.inference_spent > 6 ||
      !Number.isInteger(data.checkpoint_count) || data.checkpoint_count < 0 || data.checkpoint_count > 10) invalid()
  if (data.result) {
    if (data.result.execution_id !== id || data.result.repository_id !== repo || data.result.revision_id !== rev ||
        !Array.isArray(data.result.steps) || data.result.steps.length > 4) invalid()
    if (!['completed', 'insufficient_evidence', 'step_limit', 'provider_failure', 'validation_failure', 'invalid_decision', 'cancelled', 'timeout', 'scope_changed'].includes(data.result.status)) invalid()
    for (const step of data.result.steps) {
      if (!Number.isInteger(step.number) || step.number < 1 || step.number > 4 || !['search', 'answer', 'stop'].includes(step.operation) || !['planner', 'readiness'].includes(step.origin) || !Array.isArray(step.evidence_paths) || step.evidence_paths.length > 2 || step.evidence_paths.some(path => typeof path !== 'string')) invalid()
    }
    if (data.result.status !== 'completed' && data.result.final !== null) invalid()
    if (data.result.final) validateAnswer(data.result.final, repo, rev)
    if (data.result.status === 'completed' && (data.outcome !== 'completed' || data.result.final?.status !== 'answered')) invalid()
  }
  if (data.outcome === 'completed' && (data.status !== 'complete' || data.result?.status !== 'completed' || data.result.final?.status !== 'answered')) invalid()
  return data
}
export const terminalInvestigation = (status: string) => ['complete', 'failed', 'cancelled'].includes(status)
export const outcomeLabel = (value: string | null) => ({
  pending: 'Pending', processing: 'Processing', completed: 'Grounded answer ready',
  complete: 'Finished', failed: 'Failed', insufficient_evidence: 'Insufficient evidence',
  validation_failure: 'Answer withheld: validation failed', provider_failure: 'Provider unavailable',
  cancelled: 'Cancelled', timeout: 'Deadline reached', step_limit: 'Step limit reached',
  scope_changed: 'Repository scope changed', unknown_provider_outcome: 'Provider outcome unknown',
  invalid_decision: 'Invalid planner decision', authorization: 'Authorization changed',
  worker_lost: 'Worker recovery', validation: 'Execution validation failed'
} as Record<string, string>)[value ?? ''] ?? 'Execution ended without an accepted answer'
