import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { AuthProvider } from '../../context/AuthContext'
import { RegisterPage } from '../RegisterPage'
import { LoginPage } from '../LoginPage'

beforeEach(() => {
  localStorage.clear()
  // Only the external HTTP dependency is replaced; use the real auth context.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ access_token: 'test-token', token_type: 'bearer' })
  }))
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function mount(page: React.ReactElement) {
  render(<MemoryRouter><AuthProvider>{page}</AuthProvider></MemoryRouter>)
}

function enterCredentials(password: string, register: boolean) {
  fireEvent.change(screen.getByLabelText(/^Email$/i), { target: { value: 'person@example.com' } })
  fireEvent.change(screen.getByLabelText(/^Password$/i), { target: { value: password } })
  if (register) {
    fireEvent.change(screen.getByLabelText(/Confirm Password/i), { target: { value: password } })
  }
}

describe('registration password contract', () => {
  it.each([
    ['short12', 'Password must be at least 8 characters'],
    ['a'.repeat(73), 'Password must not exceed 72 UTF-8 bytes'],
    ['é'.repeat(37), 'Password must not exceed 72 UTF-8 bytes']
  ])('rejects invalid password without a request: %s', async (password, message) => {
    mount(<RegisterPage />)
    enterCredentials(password, true)
    fireEvent.click(screen.getByRole('button', { name: 'Register' }))
    expect(await screen.findByText(message)).toBeInTheDocument()
    expect(fetch).not.toHaveBeenCalled()
  })

  it.each(['a'.repeat(72), 'é'.repeat(36)])('accepts the byte boundary', async password => {
    mount(<RegisterPage />)
    enterCredentials(password, true)
    fireEvent.click(screen.getByRole('button', { name: 'Register' }))
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce())
    const [, options] = vi.mocked(fetch).mock.calls[0]
    expect(JSON.parse(options?.body as string).password).toBe(password)
  })
})

describe('login validation', () => {
  it('allows existing short credentials to reach server-side verification', async () => {
    mount(<LoginPage />)
    enterCredentials('short', false)
    fireEvent.click(screen.getByRole('button', { name: 'Login' }))
    await waitFor(() => expect(fetch).toHaveBeenCalledOnce())
  })

  it('rejects overlong credentials before requesting login', async () => {
    mount(<LoginPage />)
    enterCredentials('a'.repeat(73), false)
    fireEvent.click(screen.getByRole('button', { name: 'Login' }))
    expect(await screen.findByText('Password must not exceed 72 UTF-8 bytes')).toBeInTheDocument()
    expect(fetch).not.toHaveBeenCalled()
  })
})
