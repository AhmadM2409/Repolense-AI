import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { apiClient } from '../services/api'
import { useInvestigation } from './useInvestigation'
import type { InvestigationStatus } from '../types'
const scope = { id: 'id', repository_id: 'repo', revision_id: 'rev', status: 'pending' as const, outcome: null, deadline_at: '', inference_spent: 0, checkpoint_count: 0, cancel_requested: false, result: null }
beforeEach(() => { vi.useFakeTimers(); vi.spyOn(apiClient, 'investigation').mockResolvedValue(scope); vi.spyOn(apiClient, 'cancelInvestigation').mockResolvedValue({ ...scope, status: 'cancelled', cancel_requested: true }) })
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks() })
it('polls serially every two seconds and stops on terminal state', async () => {
  vi.mocked(apiClient.investigation).mockResolvedValueOnce(scope).mockResolvedValueOnce({ ...scope, status: 'processing' }).mockResolvedValue({ ...scope, status: 'complete', outcome: 'insufficient_evidence' })
  const { result } = renderHook(() => useInvestigation('repo', 'rev', 'id'))
  await act(async () => {})
  expect(apiClient.investigation).toHaveBeenCalledTimes(1)
  await act(async () => { await vi.advanceTimersByTimeAsync(1999) }); expect(apiClient.investigation).toHaveBeenCalledTimes(1)
  await act(async () => { await vi.advanceTimersByTimeAsync(1) }); expect(result.current.job?.status).toBe('processing')
  await act(async () => { await vi.advanceTimersByTimeAsync(2000) }); expect(result.current.job?.status).toBe('complete')
  await act(async () => { await vi.advanceTimersByTimeAsync(20000) }); expect(apiClient.investigation).toHaveBeenCalledTimes(3)
})
it('aborts pending status access and timers on unmount or scope change', async () => {
  const view = renderHook(({ rev }) => useInvestigation('repo', rev, 'id'), { initialProps: { rev: 'rev' } })
  await act(async () => {}); const signal = vi.mocked(apiClient.investigation).mock.calls[0][3]!
  view.rerender({ rev: 'other' }); expect(signal.aborted).toBe(true)
  await act(async () => {}); view.unmount(); const count = vi.mocked(apiClient.investigation).mock.calls.length
  await act(async () => { await vi.advanceTimersByTimeAsync(10000) }); expect(apiClient.investigation).toHaveBeenCalledTimes(count)
})
it('cancels explicitly once and does not resume polling a cancelled execution', async () => {
  const { result } = renderHook(() => useInvestigation('repo', 'rev', 'id')); await act(async () => {})
  await act(async () => { const first = result.current.cancel(); const second = result.current.cancel(); await Promise.all([first, second]) })
  expect(apiClient.cancelInvestigation).toHaveBeenCalledTimes(1); expect(result.current.job?.status).toBe('cancelled')
  await act(async () => { await vi.advanceTimersByTimeAsync(10000) }); expect(apiClient.investigation).toHaveBeenCalledTimes(1)
})
it('stops after status errors without retrying AI or polling indefinitely', async () => {
  vi.mocked(apiClient.investigation).mockRejectedValue(Object.assign(new Error('hidden'), { status: 401 }))
  const { result } = renderHook(() => useInvestigation('repo', 'rev', 'id')); await act(async () => {})
  expect(result.current.error).toMatch(/session expired/)
  await act(async () => { await vi.advanceTimersByTimeAsync(20000) }); expect(apiClient.investigation).toHaveBeenCalledTimes(1)
})
it('caps automatic polling at four minutes and permits explicit refresh', async () => {
  const { result } = renderHook(() => useInvestigation('repo', 'rev', 'id')); await act(async () => {})
  await act(async () => { await vi.advanceTimersByTimeAsync(240000) })
  expect(result.current.error).toMatch(/paused/); expect(vi.mocked(apiClient.investigation).mock.calls.length).toBeLessThanOrEqual(120)
  const count = vi.mocked(apiClient.investigation).mock.calls.length
  await act(async () => { result.current.refresh() }); expect(apiClient.investigation).toHaveBeenCalledTimes(count + 1)
})
