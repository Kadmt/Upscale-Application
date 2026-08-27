import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

const container = document.getElementById('root')!
const root = createRoot(container)
root.render(<App />)

// Register service worker on window load to enable model caching
if ('serviceWorker' in navigator) {
	window.addEventListener('load', () => {
		navigator.serviceWorker
			.register('/sw.js')
			.then((reg) => {
				console.log('ServiceWorker registered:', reg.scope)
			})
			.catch((err) => {
				console.warn('ServiceWorker registration failed:', err)
			})
	})
}
