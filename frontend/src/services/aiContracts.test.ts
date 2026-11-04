import { describe, expect, it } from 'vitest'
import { validateAnswer, validateInvestigation } from './aiContracts'
const answer = { repository_id: 'repo', revision_id: 'rev', commit_sha: 'a'.repeat(40), status: 'answered', answer: 'A supported fact.', uncertainty: '', model: 'gemini-3.5-flash-lite', citations: [{ repository_id: 'repo', revision_id: 'rev', commit_sha: 'a'.repeat(40), source_id: 'source', chunk_id: 'chunk', path: 'a.py', line_start: 1, line_end: 2, quote: 'return 1' }] }
const status = { id: 'id', repository_id: 'repo', revision_id: 'rev', status: 'complete' as const, outcome: 'completed', deadline_at: '2026-11-02T00:00:00Z', inference_spent: 4, checkpoint_count: 3, cancel_requested: false, result: { execution_id: 'id', repository_id: 'repo', revision_id: 'rev', status: 'completed', steps: [], final: answer } }
describe('scoped AI response contracts', () => {
  it('accepts server canonical responses for the exact scope', () => { expect(validateAnswer(answer, 'repo', 'rev')).toBe(answer); expect(validateInvestigation(status, 'repo', 'rev', 'id')).toBe(status) })
  it.each([{ ...answer, repository_id: 'foreign' }, { ...answer, revision_id: 'other' }, { ...answer, citations: [] }, { ...answer, commit_sha: 'branch' }, { ...answer, citations: [{ ...answer.citations[0], line_end: 0 }] }])('rejects malformed/foreign answer data', data => { expect(() => validateAnswer(data, 'repo', 'rev')).toThrow() })
  it.each([{ ...status, id: 'foreign' }, { ...status, result: null }, { ...status, inference_spent: 7 }, { ...status, result: { ...status.result, status: 'validation_failure' } }])('rejects inconsistent or foreign persisted results', data => { expect(() => validateInvestigation(data, 'repo', 'rev', 'id')).toThrow() })
})
