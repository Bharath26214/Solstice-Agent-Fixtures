# Desktop sync troubleshooting

The Solstice desktop app keeps an offline cache and syncs changes when you reconnect. Most sync problems fall into three buckets: a stale cache, a conflict, or a proxy blocking the sync endpoint.

## Stale cache

If pages show old content after reconnecting, force a cache refresh with Help → Clear local cache. This does not delete any content; the app re-downloads the workspace. On very large workspaces the first re-download can take 10 to 20 minutes.

## Conflict files

If the same page was edited offline on two devices, Solstice keeps both versions: the later save wins and the earlier one is preserved as a "(conflicted copy)" child page. Merge them manually and delete the copy. Conflicted copies are never created for online edits — those merge in real time.

## Proxy and firewall issues

The desktop app syncs over WebSocket to sync.solstice.app on port 443. Corporate proxies that strip WebSocket upgrades cause the app to fall back to polling, which is slower and can look like a hang. Ask IT to allow WebSocket traffic to *.solstice.app.

## Collecting logs

Help → Export diagnostic logs produces a zip file with the last 7 days of client logs, with content redacted. Attach it when contacting support about sync issues; it shortens diagnosis considerably.
