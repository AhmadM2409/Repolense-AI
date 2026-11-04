import { afterEach, describe, expect, it, vi } from 'vitest'
import { apiClient } from './api'

afterEach(() => { vi.unstubAllGlobals(); localStorage.clear() })

describe('API wire contracts', () => {
  it('accepts an empty 204 delete response and sends bearer credentials', async () => {
    localStorage.setItem('token', 'fixture-token')
    const fetch = vi.fn().mockResolvedValue(new Response(null, { status: 204 }))
    vi.stubGlobal('fetch', fetch)
    await expect(apiClient.deleteProfile('profile')).resolves.toBeUndefined()
    expect(fetch).toHaveBeenCalledWith('/api/profiles/profile', expect.objectContaining({
      method: 'DELETE', headers: expect.objectContaining({ Authorization: 'Bearer fixture-token' })
    }))
  })
  it('returns the status DTO without pretending it is a full review', async () => {
    const status = { review_id: 'review', status: 'pending', progress_pct: 0, error_message: null }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(status), { status: 200 })))
    await expect(apiClient.getReviewStatus('review')).resolves.toEqual(status)
  })
  it('preserves multipart upload fields and browser-generated content type', async () => {
    const form = new FormData()
    form.append('resume_file', new File(['text'], 'resume.md', { type: 'text/markdown' }))
    const fetch = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetch)
    await apiClient.createProfile(form)
    const options = fetch.mock.calls[0][1]
    expect(options.body).toBe(form)
    expect(options.headers['Content-Type']).toBeUndefined()
    expect(form.get('resume_file')).toBeInstanceOf(File)
  })
})
