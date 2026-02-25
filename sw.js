/**
 * sw.js — Service Worker GoldSignal PWA
 *
 * Stratégie de cache :
 *   - Cache-First pour les assets statiques (JS, CSS, icônes)
 *   - Network-First pour les appels API / données dynamiques
 *
 * Permet l'usage offline du calculateur terrain (page 1) une fois chargé.
 */

const CACHE_NAME = "goldsignal-v1";

// Assets à mettre en cache pour usage offline
const STATIC_ASSETS = [
  "/",
  "/manifest.json",
];

// ---------------------------------------------------------------------------
// Install : précache des assets statiques
// ---------------------------------------------------------------------------
self.addEventListener("install", (event) => {
  console.log("[SW] Install – GoldSignal v1");
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

// ---------------------------------------------------------------------------
// Activate : nettoyage des anciens caches
// ---------------------------------------------------------------------------
self.addEventListener("activate", (event) => {
  console.log("[SW] Activate – nettoyage anciens caches");
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

// ---------------------------------------------------------------------------
// Fetch : stratégie Network-First avec fallback cache
// ---------------------------------------------------------------------------
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Ignorer les requêtes non-GET et les appels WebSocket Streamlit
  if (event.request.method !== "GET") return;
  if (url.pathname.startsWith("/_stcore/stream")) return;
  if (url.protocol === "chrome-extension:") return;

  // Assets statiques → Cache-First
  if (
    url.pathname.endsWith(".js") ||
    url.pathname.endsWith(".css") ||
    url.pathname.endsWith(".png") ||
    url.pathname.endsWith(".ico") ||
    url.pathname === "/manifest.json"
  ) {
    event.respondWith(
      caches.match(event.request).then(
        (cached) => cached || fetch(event.request)
      )
    );
    return;
  }

  // Tout le reste → Network-First
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Mettre en cache la réponse fraîche
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, clone);
        });
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
