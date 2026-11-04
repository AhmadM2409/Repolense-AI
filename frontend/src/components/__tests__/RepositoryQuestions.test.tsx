import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi, describe, it, expect, beforeEach } from 'vitest'
import { RepositoryQuestions } from '../RepositoryQuestions'
import { apiClient } from '../../services/api'
vi.mock('../../services/api', () => ({ apiClient: { questionPolicy: vi.fn(), askRepository: vi.fn() } }))
describe('RepositoryQuestions', () => {
  beforeEach(() => {
    vi.mocked(apiClient.questionPolicy).mockResolvedValue({ default_provider: 'lmstudio', hosted_available: true, consent_version: 'gemini-source-processing-v1', notice: 'Google receives source excerpts' })
    vi.mocked(apiClient.askRepository).mockResolvedValue({ status: 'answered', answer: 'Source fact', uncertainty: '', model: 'gemini-3.5-flash-lite', revision_id: 'r', citations: [{ repository_id: "repo", revision_id: "r", source_id: "source", chunk_id: "chunk", commit_sha: "a".repeat(40), path: "guard.py", line_start: 1, line_end: 2, quote: "return 1" }] })
  })
  it('requires visible consent for each hosted request', async () => {
    render(<RepositoryQuestions repositoryId="repo" revisionId="r" />)
    await screen.findByText('Google Gemini (hosted)')
    fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'Explain guard' } })
    fireEvent.change(screen.getByLabelText('Provider'), { target: { value: 'gemini' } })
    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled()
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }))
    await waitFor(() => expect(apiClient.askRepository).toHaveBeenCalledWith('repo', 'r', expect.objectContaining({ generation_provider: 'gemini', hosted_processing_consent: 'gemini-source-processing-v1' }), expect.any(AbortSignal)))
    await screen.findByText('Source fact')
    expect(screen.getByRole('checkbox')).not.toBeChecked()
  })
})
