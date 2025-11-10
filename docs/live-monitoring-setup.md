# Live Monitoring Dashboard Pattern

**Last updated:** 2025-11-10

## Pattern

Monitor long-running processes with a browser dashboard that auto-refreshes:

```
background_process → data_file.json ← HTML dashboard (via HTTP)
```

Components:
1. **Data writer** (Python script, cron job, shell loop) - writes JSON periodically
2. **Dashboard** (HTML + JavaScript) - fetches and displays JSON
3. **HTTP server** (required for CORS) - serves both files from localhost

## Critical Rule: HTTP Required

Browsers block `fetch()` from `file://` to local files. Opening `dashboard.html` directly will fail silently.

**WRONG:**
```bash
open monitor_dashboard.html  # file:// protocol, fetch blocked
```

**CORRECT:**
```bash
python3 -m http.server 8000 > /dev/null 2>&1 &
open http://localhost:8000/monitor_dashboard.html
```

Same-origin policy allows `http://localhost:8000/dashboard.html` to fetch `http://localhost:8000/data.json`.

## Implementation

**1. Data writer (Python example):**
```python
import json, time

while True:
    data = collect_metrics()  # Your logic here
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=2)
    time.sleep(10)
```

**2. Dashboard (HTML + JS):**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Monitor</title>
    <style>
        body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }
        /* Your styles */
    </style>
</head>
<body>
    <div id="content"></div>
    <script>
        async function loadData() {
            const response = await fetch('data.json?t=' + Date.now());
            const data = await response.json();
            document.getElementById('content').innerHTML = renderData(data);
        }
        function renderData(data) { return JSON.stringify(data, null, 2); }  // Customize
        setInterval(loadData, 10000);  // Refresh every 10s
        loadData();  // Initial load
    </script>
</body>
</html>
```

**3. Launch:**
```bash
python data_writer.py &          # Start data collection
python3 -m http.server 8000 &    # Serve files
open http://localhost:8000/dashboard.html
```

## Design Rules

**No emojis.** UTF-8 encoding breaks across browsers/terminals. Use ASCII symbols: `|`, `*`, `+`, `-`, or unicode box characters.

**Cache bust.** Add timestamp to fetch URL: `fetch('data.json?t=' + Date.now())` prevents stale cached data.

**Gitignore generated data.** Add `data.json` or `*.json` to `.gitignore` if file is ephemeral.

## Troubleshooting

**Port already in use:**
```bash
lsof -ti:8000 | xargs kill -9   # Kill existing server
python3 -m http.server 8001 &   # Use different port
open http://localhost:8001/dashboard.html
```

**Dashboard shows stale data:** Check cache-bust parameter (`?t=`) is present in fetch URL.

**Data writer not updating:** Verify process is running (`ps aux | grep writer`), check file timestamps (`ls -lt data.json`).

## Why This Works

- **Persistent view** - browser tab stays open, no truncation
- **Decoupled** - data source, HTTP server, dashboard are independent processes
- **Universal** - works for any data (logs, metrics, experiment results, build status)
- **Zero dependencies** - Python stdlib + vanilla HTML/JS
- **Portable** - works on any machine with Python and a browser
