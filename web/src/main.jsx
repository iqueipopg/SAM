// Copyright 2026 Ignacio Queipo de Llano Pérez-Gascón and the SAM authors
// SPDX-License-Identifier: Apache-2.0
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
