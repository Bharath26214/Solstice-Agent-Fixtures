# Integrations

Solstice integrates with Slack, GitHub, Jira, Figma, and Google Drive on all paid plans. The Free plan includes the Slack integration only.

## Slack

Install from Settings → Integrations → Slack. Once connected you can unfurl Solstice links in Slack, receive task notifications in channels, and create tasks from Slack messages with the /solstice command. The integration needs to be installed by both a Solstice admin and a Slack workspace admin.

## GitHub

The GitHub integration links pull requests to Solstice tasks. Mention a task ID (like SOL-142) in a PR title or branch name and the task moves to "In review" when the PR opens and "Done" when it merges. Configure which repositories are linked under Settings → Integrations → GitHub.

## Jira

The Jira integration is one-way: it imports Jira issues into Solstice as read-only cards for cross-team visibility. Two-way sync is on the roadmap but not available today. Import runs every 15 minutes.

## Building your own

Use the REST API and webhooks to build custom integrations. Webhook payloads are signed (see the API documentation) and deliveries are retried with exponential backoff for 24 hours.
