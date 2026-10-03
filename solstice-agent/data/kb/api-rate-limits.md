# API rate limits

The Solstice REST API is available on every plan. Rate limits are applied per API key: the Free plan allows 60 requests per minute, Starter allows 300, Pro allows 600, and Enterprise allows 3,000. Burst traffic up to 2x the limit is tolerated for 10 seconds before requests are rejected.

## Handling 429 responses

When you exceed your limit the API returns HTTP 429 with a Retry-After header giving the number of seconds to wait. Honor Retry-After rather than retrying immediately; repeated violations within an hour temporarily halve your limit.

## API keys

Workspace admins create and revoke API keys under Settings → Developers. Keys are shown once at creation. We recommend rotating keys every 90 days; keys unused for 180 days are disabled automatically. Each key can be scoped read-only or read-write, and scoped to specific projects.

## Webhooks

Webhook deliveries are signed with an HMAC-SHA256 signature in the X-Solstice-Signature header. Verify it using your webhook secret before trusting the payload. Deliveries are retried with exponential backoff for up to 24 hours; endpoints failing for 7 consecutive days are disabled.

## API overage

Usage-based overage applies only to the Data Export API on the Pro plan: exports beyond 50 GB per month are billed at $0.02 per GB. Overage fees appear as separate line items on the next invoice and are not refundable.
