# Deployment (run the app continuously for the 7-day live test)

Requirements: always-on host (free tiers that sleep will fail the test), server region where Binance market data
works (Europe or Asia; some US cloud IPs are blocked), and **no code changes once the test starts**.

## Option 1: small VPS (recommended; about 4-6 USD/month: Hetzner, DigitalOcean, AWS Lightsail, etc.)
1. Create an Ubuntu 22.04 server in a Europe or Asia region. Note its IP address.
2. SSH in:  `ssh root@<server-ip>`
3. Install Docker:  `curl -fsSL https://get.docker.com | sh`
4. Get the code (the `models/` folder MUST be committed to GitHub):
   ```bash
   git clone https://github.com/<you>/sol-ai-indicator.git && cd sol-ai-indicator
   ```
5. Build and run (auto-restarts after crashes and reboots; the volume keeps the signal log):
   ```bash
   docker build -t sol-indicator .
   docker run -d --name sol --restart always -p 80:8000 -v sol_data:/app/data sol-indicator
   ```
6. Open `http://<server-ip>/` in a browser. If it does not load, open port 80 in the provider's firewall.
7. Check health:  `http://<server-ip>/api/health`  (`last_close_age_sec` should stay under about 1000 for 15m candles).
8. Logs:  `docker logs -f sol`

## Dress rehearsal (do at least 7 days before the deadline)
- Run it for a full week. Check every day: health endpoint, signal count, scorecard.
- Fix problems BEFORE the freeze. After the freeze, change nothing.

## Update the model later (before the freeze only)
```bash
git pull && docker build -t sol-indicator . && docker rm -f sol && \
docker run -d --name sol --restart always -p 80:8000 -v sol_data:/app/data sol-indicator
```

## Common failures
| Symptom | Cause / fix |
|---|---|
| Page loads, chart empty | Server cannot reach Binance: wrong region. Move the server. |
| `models/config.json not found` | `models/` was not pushed to GitHub. Commit and push it. |
| Signals stop appearing | Check `/api/health`; `docker logs sol` for "stream error". |
| Lost signal history after redeploy | Run with the `-v sol_data:/app/data` volume. |
