import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate, useParams } from 'react-router-dom'
import { useAuth } from './context/AuthContext'
import { NavBar } from './components/NavBar'
import { LoginPage } from './pages/LoginPage'
import { RegisterPage } from './pages/RegisterPage'
import { JobFitAnalyzerPage } from './pages/JobFitAnalyzerPage'
import { RepositoryWorkspacePage } from './pages/RepositoryWorkspacePage'
import { RepositoryAnalysisPage } from './pages/RepositoryAnalysisPage'
import { RepositoryIngestionPage } from './pages/RepositoryIngestionPage'
import { DashboardPage } from './pages/DashboardPage'
import { NewProfilePage } from './pages/NewProfilePage'
import { ReviewHistoryPage } from './pages/ReviewHistoryPage'

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}

function ReviewJobLink() {
  const { reviewId = '' } = useParams()
  return <Navigate to={'/analyses/' + reviewId} replace />
}

function App() {
  const { user, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <>
      <NavBar />
      <Routes>
        <Route
          path="/"
          element={user ? <Navigate to="/dashboard" replace /> : <Navigate to="/login" replace />}
        />
        <Route path="/analyses/:jobId" element={<ProtectedRoute><RepositoryAnalysisPage /></ProtectedRoute>} />
        <Route path="/career" element={<ProtectedRoute><JobFitAnalyzerPage key={user?.id} /></ProtectedRoute>} />
        <Route path="/workspace" element={<ProtectedRoute><RepositoryWorkspacePage key={user?.id} /></ProtectedRoute>} />
        <Route path="/repositories/new" element={<ProtectedRoute><RepositoryIngestionPage /></ProtectedRoute>} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/profiles/new"
          element={
            <ProtectedRoute>
              <NewProfilePage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/reviews"
          element={
            <ProtectedRoute>
              <ReviewHistoryPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/reviews/:reviewId"
          element={
            <ProtectedRoute>
              <ReviewJobLink />
            </ProtectedRoute>
          }
        />
      </Routes>
    </>
  )
}

export default App
