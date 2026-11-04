import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { RepositoryIngestionPage } from '../RepositoryIngestionPage'

const profileId = '11111111-1111-4111-8111-111111111111'
const pending = { id: 'job', profile_id: profileId, status: 'pending', requested_repository: 'example/project',
  revision_id: null, error_message: null, ingestion_result: null }
const complete = { ...pending, status: 'complete', revision_id: 'revision', ingestion_result: {
  commit_sha: 'a'.repeat(40), file_count: 1, chunk_count: 1, deduplicated: false, skipped: [],
  files: [{ id: 'source', path: 'README.md', content_hash: 'b'.repeat(64), chunk_count: 1 }]
} }
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })
function show() {
  render(<MemoryRouter><RepositoryIngestionPage /></MemoryRouter>)
  fireEvent.change(screen.getByLabelText('Public GitHub repository'), { target: { value: 'example/project' } })
}
async function submit() {
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Ingest repository' })) })
}
beforeEach(() => { vi.useFakeTimers(); localStorage.setItem('token', 'user-fixture-token') })
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); localStorage.clear() })

describe('repository ingestion wire flow', () => {
  it('creates an owned profile, polls truthful status and previews exact source text', async () => {
    const fetch = vi.fn()
      .mockResolvedValueOnce(response({ id: profileId }))
      .mockResolvedValueOnce(response(pending, 202))
      .mockResolvedValueOnce(response(complete))
      .mockResolvedValueOnce(response({ source: { path: 'README.md', text: '<script>fixture</script>', content_hash: 'b'.repeat(64) }, chunks: [] }))
    vi.stubGlobal('fetch', fetch)
    show(); await submit()
    expect(screen.getByRole('status')).toHaveTextContent('Queued')
    await act(async () => { await vi.advanceTimersByTimeAsync(1000) })
    expect(screen.getByText('Ingestion status: complete')).toBeInTheDocument()
    expect(screen.getByText('a'.repeat(40))).toBeInTheDocument()
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'README.md' })) })
    expect(screen.getByText('<script>fixture</script>')).toBeInTheDocument()
    expect(document.querySelector('script')).toBeNull()
    const submission = fetch.mock.calls.find(call => call[0] === '/api/repository-ingestions')
    expect(JSON.parse(submission![1].body)).toEqual({ profile_id: profileId, repository: 'example/project' })
    expect(submission![1].headers.Authorization).toBe('Bearer user-fixture-token')
    expect(JSON.stringify(fetch.mock.calls)).not.toContain('github_token')
  })

  it('shows recorded failure without a completed state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(response({ id: profileId }))
      .mockResolvedValueOnce(response({ ...pending, status: 'failed', error_message: 'GitHub rate limit reached; retry later' }, 202)))
    show(); await submit()
    expect(screen.getByRole('alert')).toHaveTextContent('GitHub rate limit')
    expect(screen.queryByText('Ingestion status: complete')).toBeNull()
    expect(screen.getByRole('button', { name: 'Ingest repository' })).toBeEnabled()
  })

  it('shows HTTP errors and retains user input for retry', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({ detail: 'Profile not found' }, 404)))
    show()
    fireEvent.change(screen.getByLabelText('Profile ID (optional)'), { target: { value: profileId } })
    await submit()
    expect(screen.getByRole('alert')).toHaveTextContent('Profile not found')
    expect(screen.getByLabelText('Public GitHub repository')).toHaveValue('example/project')
  })

  it('stops polling on unmount', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(response({ id: profileId })).mockResolvedValueOnce(response(pending, 202))
    vi.stubGlobal('fetch', fetch)
    show(); await submit(); cleanup()
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    expect(fetch).toHaveBeenCalledTimes(2)
  })

  it('reuses the selected profile when ingesting again', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(response({ id: profileId }))
      .mockResolvedValueOnce(response(complete, 202))
      .mockResolvedValueOnce(response({ ...complete, ingestion_result: { ...complete.ingestion_result, deduplicated: true } }, 202))
    vi.stubGlobal('fetch', fetch)
    show(); await submit(); await submit()
    expect(fetch.mock.calls.filter(call => call[0] === '/api/profiles')).toHaveLength(1)
    expect(screen.getByText(/Existing verified revision reused/)).toBeInTheDocument()
  })
})

it('starts deterministic analysis from the completed ingestion', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(response({ id: profileId }))
    .mockResolvedValueOnce(response(complete, 202))
    .mockResolvedValueOnce(response({ id: 'analysis', status: 'pending' }, 202))
  vi.stubGlobal('fetch', fetch)
  show(); await submit()
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Analyze this revision' })) })
  const request = fetch.mock.calls.find(call => call[0] === '/api/analyses')
  expect(JSON.parse(request![1].body)).toEqual({ ingestion_id: 'job' })
  expect(request![1].headers.Authorization).toBe('Bearer user-fixture-token')
})
