import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RepositoryWorkspacePage } from '../RepositoryWorkspacePage'

const repo = '11111111-1111-4111-8111-111111111111', rev = '22222222-2222-4222-8222-222222222222', id = '33333333-3333-4333-8333-333333333333'
const sha = 'a'.repeat(40)
const citation = { repository_id: repo, revision_id: rev, source_id: 'source', chunk_id: 'chunk', commit_sha: sha, path: 'src/guard.py', line_start: 1, line_end: 2, quote: '<script>untrusted</script>\nreturn 1' }
const answer = { repository_id: repo, revision_id: rev, commit_sha: sha, status: 'answered', answer: 'guard returns 1.', uncertainty: '', model: 'gemini-3.5-flash-lite', citations: [citation] }
const policy = { default_provider: 'gemini', hosted_available: true, durable_investigations_available: true, consent_version: 'gemini-source-processing-v1', notice: 'Google receives this question and source excerpts.' }
const terminal = { id, repository_id: repo, revision_id: rev, status: 'complete', outcome: 'completed', deadline_at: '2026-11-02T00:00:00Z', inference_spent: 4, checkpoint_count: 3, cancel_requested: false, result: { execution_id: id, repository_id: repo, revision_id: rev, status: 'completed', final: answer, steps: [{ number: 1, operation: 'search', origin: 'planner', evidence_paths: ['src/guard.py'] }] } }
const row = { id, objective: 'Explain the saved guard', status: 'complete', outcome: 'completed', created_at: '2026-11-02T00:00:00Z', completed_at: '2026-11-02T00:00:10Z' }
const response = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status })
let askStatus = 200, askBody: unknown = answer, history = false, empty = false, noRevisions = false, disabled = false
let fetcher: ReturnType<typeof vi.fn>
function show(path = '/workspace') { return render(<MemoryRouter initialEntries={[path]}><RepositoryWorkspacePage /></MemoryRouter>) }
async function select() {
  await screen.findByRole('option', { name: 'fixture/repository' })
  fireEvent.change(screen.getByLabelText('Repository'), { target: { value: repo } })
  await screen.findByRole('option', { name: sha })
  fireEvent.change(screen.getByLabelText('Immutable revision'), { target: { value: rev } })
  await screen.findByRole('heading', { name: 'Ask about this revision' })
}
beforeEach(() => {
  askStatus = 200; askBody = answer; history = false; empty = false; noRevisions = false; disabled = false
  localStorage.setItem('token', 'synthetic-session')
  fetcher = vi.fn(async (url: string, options?: RequestInit) => {
    const path = url.split('?')[0]
    if (path === '/api/repositories') return response({ items: empty ? [] : [{ id: repo, full_name: 'fixture/repository', provider: 'github' }], next_offset: null })
    if (path.endsWith('/revisions')) return response({ items: noRevisions ? [] : [{ id: rev, repository_id: repo, commit_sha: sha }], next_offset: null })
    if (path.endsWith('/question-policy')) return response({ ...policy, durable_investigations_available: !disabled })
    if (path.endsWith('/questions')) return response(askBody, askStatus)
    if (path.endsWith('/durable-investigations') && options?.method === 'POST') { history = true; return response({ id, status: 'pending' }, 202) }
    if (path.endsWith('/durable-investigations')) return response({ items: history ? [row] : [], next_offset: null })
    if (path.endsWith('/cancel')) return response({ ...terminal, status: 'cancelled', outcome: null, result: null, cancel_requested: true })
    return response(terminal)
  })
  vi.stubGlobal('fetch', fetcher)
})
afterEach(() => { vi.unstubAllGlobals(); localStorage.clear() })

describe('Repository AI workspace', () => {
  it('requires exact repository/revision before rendering either submission form', async () => {
    show(); expect(screen.queryByLabelText('Question')).toBeNull(); await select()
    expect(screen.getByText('Active commit '+sha)).toBeInTheDocument()
    expect(fetcher.mock.calls.some(c => c[1]?.method === 'POST')).toBe(false)
  })
  it('handles an empty catalog and a repository without revisions', async () => {
    empty = true; const view = show(); await screen.findByText('Your repository workspace starts here'); view.unmount()
    empty = false; noRevisions = true; show(); await screen.findByRole('option', { name: 'fixture/repository' })
    fireEvent.change(screen.getByLabelText('Repository'), { target: { value: repo } })
    await screen.findByText('No available revision yet'); expect(screen.queryByLabelText('Question')).toBeNull()
  })
  it('submits Q&A only on explicit consent, once, with session and scope; renders inert canonical evidence', async () => {
    show(); await select(); await screen.findByText(policy.notice)
    fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'Explain guard' } })
    const submit = screen.getAllByRole('button', { name: 'Ask' }).find(b => b.getAttribute('type') === 'submit')!
    expect(submit).toBeDisabled(); fireEvent.click(screen.getByRole('checkbox'))
    await act(async () => { fireEvent.click(submit); fireEvent.click(submit) })
    await screen.findByText('guard returns 1.')
    const calls = fetcher.mock.calls.filter(c => c[0].endsWith('/questions'))
    expect(calls).toHaveLength(1)
    expect(calls[0][1]?.headers).toMatchObject({ Authorization: 'Bearer synthetic-session' })
    expect(JSON.parse(calls[0][1]?.body as string)).toMatchObject({ question: 'Explain guard', generation_provider: 'gemini', hosted_processing_consent: 'gemini-source-processing-v1' })
    fireEvent.click(screen.getByText('src/guard.py:1–2'))
    expect(screen.getByText(/<script>untrusted<\/script>/)).toBeInTheDocument(); expect(document.querySelector('script')).toBeNull()
    expect(screen.getByRole('checkbox')).not.toBeChecked()
  })
  it.each([['validation_failure', 'Answer withheld: validation failed'], ['provider_failure', 'Provider unavailable'], ['insufficient_evidence', 'Insufficient evidence']])('does not publish rejected %s prose', async (status, label) => {
    askBody = { ...answer, status, answer: 'UNACCEPTED', uncertainty: 'More evidence is needed.', citations: [] }
    show(); await select(); await screen.findByText(policy.notice)
    fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'Explain guard' } }); fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.submit(screen.getByLabelText('Question').closest('form')!)
    await screen.findByText(label); expect(screen.queryByText('UNACCEPTED')).toBeNull()
  })
  it.each([401, 502, 503])('shows truthful HTTP %s errors without a returned answer', async status => {
    askStatus = status; askBody = { detail: 'PRIVATE_PROVIDER_DIAGNOSTIC' }
    show(); await select(); await screen.findByText(policy.notice)
    fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'Explain guard' } }); fireEvent.click(screen.getByRole('checkbox')); fireEvent.submit(screen.getByLabelText('Question').closest('form')!)
    await screen.findByRole('alert'); expect(screen.queryByText('PRIVATE_PROVIDER_DIAGNOSTIC')).toBeNull(); expect(screen.queryByText('guard returns 1.')).toBeNull()
    if (status === 401) expect(screen.getByRole('alert')).toHaveTextContent('session expired')
  })
  it('starts a durable investigation once and can reopen its server history without another inference request', async () => {
    show(); await select(); fireEvent.click(screen.getByRole('button', { name: 'Investigate' })); await screen.findByText(policy.notice)
    fireEvent.change(screen.getByLabelText('Investigation objective'), { target: { value: 'Trace guard' } })
    expect(screen.getByRole('button', { name: 'Start investigation' })).toBeDisabled()
    fireEvent.click(screen.getByRole('checkbox'))
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Start investigation' })); fireEvent.submit(screen.getByLabelText('Investigation objective').closest('form')!) })
    await screen.findByText('guard returns 1.')
    await screen.findByRole('button', { name: /Explain the saved guard/ })
    fireEvent.click(screen.getByRole('button', { name: /Explain the saved guard/ }))
    await waitFor(() => expect(screen.getByRole('region', { name: 'Investigation status' })).toHaveTextContent(id))
    expect(fetcher.mock.calls.filter(c => c[0].endsWith('/durable-investigations') && c[1]?.method === 'POST')).toHaveLength(1)
  })
  it('reloads a URL investigation using reads only and shows committed steps', async () => {
    history = true; show(`/workspace?repository=${repo}&revision=${rev}&investigation=${id}`)
    await screen.findByText('guard returns 1.'); expect(screen.getByText('Saved execution steps')).toBeInTheDocument()
    expect(fetcher.mock.calls.some(c => c[1]?.method === 'POST')).toBe(false)
  })
  it('shows disabled investigations without silently enabling flags', async () => {
    disabled = true; show(); await select(); fireEvent.click(screen.getByRole('button', { name: 'Investigate' }))
    await screen.findByText(/Durable investigations are not enabled/); expect(screen.getByRole('button', { name: 'Start investigation' })).toBeDisabled()
  })
  it('clears question and prior answers when the revision is deselected', async () => {
    show(); await select(); fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'Old question' } })
    fireEvent.change(screen.getByLabelText('Immutable revision'), { target: { value: '' } })
    expect(screen.queryByLabelText('Question')).toBeNull(); expect(screen.queryByText('guard returns 1.')).toBeNull()
  })
})
