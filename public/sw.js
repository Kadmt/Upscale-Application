const CACHE_NAME = 'models-cache-v1';

self.addEventListener('install', (e) => {
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(self.clients.claim());
});

// Cache-on-demand for any request under /models/
self.addEventListener('fetch', (e) => {
  try {
    const url = new URL(e.request.url);
    if (url.pathname.startsWith('/models/')) {
      e.respondWith(
        caches.open(CACHE_NAME).then(async (cache) => {
          const cached = await cache.match(e.request);
          if (cached) return cached;
          const resp = await fetch(e.request);
          if (resp && resp.ok && e.request.method === 'GET') {
            cache.put(e.request, resp.clone()).catch(() => {});
          }
          return resp;
        })
      );
    }
  } catch (err) {
    // ignore URL parse errors or cross-origin
  }
});

self.addEventListener('message', (e) => {
  const msg = e.data || {};
  if (msg && msg.action === 'clearModelsCache') {
    caches.delete(CACHE_NAME);
  }
});
