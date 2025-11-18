# Walmart Social Listener - Windows Startup Script
Write-Host "🚀 Starting Walmart Social Listener..." -ForegroundColor Green

# Start FastAPI backend
Write-Host "🐍 Starting FastAPI backend on port 8000..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    $env:PYTHONPATH = $using:PWD
    py -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
}

# Start Node.js server
Write-Host "🟢 Starting Node.js server on port 3001..." -ForegroundColor Yellow
$nodeJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    cd server
    npm start
}

# Start React frontend
Write-Host "⚛️ Starting React frontend on port 5173..." -ForegroundColor Yellow
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    cd frontend
    npm run dev
}

Write-Host ""
Write-Host "✅ All servers started in background!" -ForegroundColor Green
Write-Host "📊 FastAPI Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "🔄 Node.js Server: http://localhost:3001" -ForegroundColor Cyan
Write-Host "⚛️ React Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "To view job output, run: Get-Job | Receive-Job" -ForegroundColor Gray
Write-Host "To stop all servers, run: Get-Job | Stop-Job; Get-Job | Remove-Job" -ForegroundColor Gray
Write-Host ""

# Keep script running and show job status
while ($true) {
    Start-Sleep -Seconds 5
    $jobs = Get-Job
    $running = ($jobs | Where-Object { $_.State -eq 'Running' }).Count
    if ($running -eq 0) {
        Write-Host "⚠️ All jobs have stopped." -ForegroundColor Red
        break
    }
}

