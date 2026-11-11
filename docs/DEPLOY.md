# Deployment Guide (SOL AI Trading Indicator)

To evaluate your live indicator (Option B) for the 7-day live market test and submit a live URL to your instructor, you have two primary deployment paths:

---

## Method 1: Free Cloud Deployment via Render (Recommended for Student Projects)

**Why Render:** 
- 100% Free tier.
- Gives you an official public HTTPS URL (e.g. `https://sol-ai-indicator.onrender.com`) that you can submit to your teacher and paste into your report.
- Zero server maintenance: auto-builds directly from your GitHub repo.
- Supported in Europe/Asia (Frankfurt or Singapore) so Binance WebSocket streams work without cloud geo-blocking.

### Steps to Deploy in 2 Minutes:
1. Make sure your latest changes are pushed to GitHub:
   ```bash
   git push origin main
   ```
2. Go to [https://render.com](https://render.com) and sign up / log in with your GitHub account.
3. Click **"New +"** &rarr; **"Web Service"**.
4. Connect your `sol-ai-indicator` GitHub repository.
5. Fill in the settings:
   - **Name:** `sol-ai-indicator`
   - **Region:** **Frankfurt (EU)** or **Singapore (Asia)** *(Important: Binance blocks some US cloud IPs, so choose Frankfurt or Singapore)*.
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.server.app:app --host 0.0.0.0 --port $PORT`
   - **Plan Type:** `Free`
6. Click **"Create Web Service"**.
7. Render will build and deploy the app. Within ~2 minutes, your live URL will appear at the top (e.g. `https://sol-ai-indicator.onrender.com`).
8. Open the URL in any browser or on your phone to verify the live chart and indicator!

---

## Method 2: Deployment on Railway (Alternative Cloud Host)

1. Push your code to GitHub.
2. Go to [https://railway.app](https://railway.app) and log in with GitHub.
3. Click **"New Project"** &rarr; **"Deploy from GitHub repo"**.
4. Select `sol-ai-indicator`.
5. Railway automatically detects `Procfile` / `Dockerfile` and deploys.
6. Under Settings &rarr; Networking, click **"Generate Domain"** to get your public HTTPS link.

---

## Method 3: Cloud VPS with Docker (Hetzner, DigitalOcean, Lightsail - $4-5/month)

1. Create an Ubuntu 22.04 VPS in Europe (Frankfurt/London) or Asia (Singapore).
2. SSH into your server:
   ```bash
   ssh root@<server-ip>
   ```
3. Install Docker:
   ```bash
   curl -fsSL https://get.docker.com | sh
   ```
4. Clone your repository:
   ```bash
   git clone https://github.com/<your-username>/sol-ai-indicator.git && cd sol-ai-indicator
   ```
5. Build and launch with persistent storage:
   ```bash
   docker build -t sol-indicator .
   docker run -d --name sol --restart always -p 80:8000 -v sol_data:/app/data sol-indicator
   ```
6. Open `http://<server-ip>/` in your browser.
7. Verify health: `http://<server-ip>/api/health`.

---

## Method 4: Instant Temporary Public Link from Your Local Machine (ngrok)

If you need a shareable public HTTPS link within 30 seconds for an immediate demonstration to your teacher without deploying to cloud:
1. Keep your local server running:
   ```powershell
   python -m uvicorn src.server.app:app --port 8000
   ```
2. In a second terminal, run `ngrok` (download from [ngrok.com](https://ngrok.com)):
   ```powershell
   ngrok http 8000
   ```
3. Copy the generated `https://xxxx.ngrok-free.app` URL and send it to your teacher.
