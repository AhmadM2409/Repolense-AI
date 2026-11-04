import React from 'react'
import { useNavigate } from 'react-router-dom'
import { LogOut, BarChart3 } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

export const NavBar: React.FC = () => {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  if (!user) return null

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex min-h-16 flex-wrap justify-between items-center gap-3 py-3">
          <button type="button" aria-label="RepoLens AI dashboard" className="flex items-center gap-2 focus-visible:outline-blue-600" onClick={() => navigate('/dashboard')}>
            <BarChart3 className="w-6 h-6 text-blue-600" />
            <span className="text-xl font-bold text-gray-900">RepoLens AI</span>
          </button>

          <div className="flex flex-wrap items-center gap-4">
            <button
              onClick={() => navigate('/dashboard')}
              className="text-gray-700 hover:text-gray-900 font-medium"
            >
              Dashboard
            </button>
            <button onClick={() => navigate('/career')} className="text-blue-700 hover:text-blue-900 font-medium">Job Fit Analyzer</button>
            <button onClick={() => navigate('/workspace')} className="text-blue-700 hover:text-blue-900 font-medium">Repository AI</button>
            <button
              onClick={() => navigate('/reviews')}
              className="text-gray-700 hover:text-gray-900 font-medium"
            >
              Reviews
            </button>

            <div className="flex items-center gap-3 pl-6 border-l border-gray-200">
              <span className="text-sm text-gray-600">{user.email}</span>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
                title="Logout"
              >
                <LogOut className="w-5 h-5 text-gray-600" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
