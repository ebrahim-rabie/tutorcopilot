#Requires -Version 5.1
<#
  try_tutor.ps1 - Walk through every TutorCopilot endpoint against a running server.

  Usage:
    .\scripts\try_tutor.ps1                       # default question
    .\scripts\try_tutor.ps1 -Base http://127.0.0.1:8000 -Question "Explain RRF with an example"

  Prerequisites:
    1. .env has TC_LLM_API_KEY filled in
    2. Server running:  python -m uvicorn app.main:app --port 8000
#>
param(
    [string]$Base = "http://127.0.0.1:8000",
    [string]$Question = "When should I fine-tune instead of using RAG?"
)

$ErrorActionPreference = "Stop"
$sep = "=" * 60

function Section($n, $t) { Write-Host "`n$sep" -ForegroundColor DarkGray; Write-Host " [$n] $t" -ForegroundColor Cyan; Write-Host $sep -ForegroundColor DarkGray }
function Fail($msg) { Write-Host "FAILED: $msg" -ForegroundColor Red; exit 1 }

# --- 0. Server up? -------------------------------------------------------
Section 0 "Server health"
try {
    $health = Invoke-RestMethod "$Base/health" -TimeoutSec 15
} catch {
    Fail "server not reachable at $Base`nStart it with:  python -m uvicorn app.main:app --port 8000"
}
$health | ConvertTo-Json -Compress
if (-not $health.llm_configured) {
    Write-Host "`nLLM not configured. Put your Groq key in .env (line: TC_LLM_API_KEY=gsk_...) and restart the server." -ForegroundColor Yellow
    exit 1
}
Write-Host "index chunks: $($health.index_chunks) | llm configured: yes" -ForegroundColor Green

# --- 1. Single-shot RAG --------------------------------------------------
Section 1 "POST /ask  (single-shot RAG, no memory)"
$body = @{ question = $Question } | ConvertTo-Json -Compress
try {
    $ask = Invoke-RestMethod -Method Post -Uri "$Base/ask" -ContentType "application/json" -Body $body -TimeoutSec 120
    Write-Host "answer:" -ForegroundColor Green
    Write-Host $ask.answer
    Write-Host "`ncitations: $($ask.citations -join ', ') | confidence: $($ask.confidence) | $($ask.latency_ms)ms | trace: $($ask.trace_id)"
} catch {
    Write-Host "note: /ask failed -> $($_.Exception.Message)" -ForegroundColor Yellow
}

# --- 2. Agent turn 1 -----------------------------------------------------
Section 2 "POST /agent/ask  (multi-step agent with tools)"
$agentBody = @{ question = $Question } | ConvertTo-Json -Compress
try {
    $a1 = Invoke-RestMethod -Method Post -Uri "$Base/agent/ask" -ContentType "application/json" -Body $agentBody -TimeoutSec 180
} catch {
    Fail "agent/ask -> $($_.Exception.Message)"
}
Write-Host "answer:" -ForegroundColor Green
Write-Host $a1.answer
Write-Host "`nconversation_id: $($a1.conversation_id)"
Write-Host "tools used:      $(if ($a1.tools_called.Count) { $a1.tools_called -join ', ' } else { '(none)' })"
Write-Host "steps: $($a1.steps.Count) | $($a1.total_latency_ms)ms | success: $($a1.success)"
Write-Host "trace_id: $($a1.trace_id)"
if ($a1.error) { Write-Host "error: $($a1.error)" -ForegroundColor Yellow }

# --- 3. Agent turn 2 (same conversation -> memory) -----------------------
Section 3 "POST /agent/ask  again with conversation_id (memory carries over)"
$followBody = @{ question = "Now explain it more briefly."; conversation_id = $a1.conversation_id } | ConvertTo-Json -Compress
try {
    $a2 = Invoke-RestMethod -Method Post -Uri "$Base/agent/ask" -ContentType "application/json" -Body $followBody -TimeoutSec 180
    Write-Host $a2.answer
    Write-Host "`nsame conversation: $($a2.conversation_id -eq $a1.conversation_id) | tools: $($a2.tools_called -join ', ')"
} catch {
    Write-Host "note: follow-up failed -> $($_.Exception.Message)" -ForegroundColor Yellow
}

# --- 4. Conversation history --------------------------------------------
Section 4 "GET /agent/conversations/{id}"
try {
    $conv = Invoke-RestMethod "$Base/agent/conversations/$($a1.conversation_id)" -TimeoutSec 30
    $conv | ConvertTo-Json -Depth 6 -Compress
} catch {
    Write-Host "note: $($_.Exception.Message)" -ForegroundColor Yellow
}

# --- 5. Weak topics ------------------------------------------------------
Section 5 "GET /agent/memory/weak-topics  (what the learner struggles with)"
try {
    $weak = Invoke-RestMethod "$Base/agent/memory/weak-topics" -TimeoutSec 30
    $weak | ConvertTo-Json -Depth 5 -Compress
} catch {
    Write-Host "note: $($_.Exception.Message)" -ForegroundColor Yellow
}

# --- 6. Trace ------------------------------------------------------------
Section 6 "GET /traces/{trace_id}  (full observability record)"
try {
    $trace = Invoke-RestMethod "$Base/traces/$($a1.trace_id)" -TimeoutSec 30
    Write-Host "trace ok: $(($trace | Get-Member -MemberType NoteProperty).Name -join ', ')"
} catch {
    Write-Host "note: $($_.Exception.Message)" -ForegroundColor Yellow
}

# --- 7. SSE streaming ----------------------------------------------------
Section 7 "POST /agent/stream  (live SSE: thought -> tool -> final)"
Write-Host "(streaming raw events, Ctrl+C to stop)" -ForegroundColor DarkGray
$sseBody = @{ question = "What is the markdown chunking strategy used here?" } | ConvertTo-Json -Compress
$sseFile = New-TemporaryFile
[IO.File]::WriteAllText($sseFile.FullName, $sseBody, [Text.UTF8Encoding]::new($false))
curl.exe -sS -N -X POST "$Base/agent/stream" -H "Content-Type: application/json" --data-binary "@$($sseFile.FullName)" --max-time 180
if ($LASTEXITCODE -ne 0) { Write-Host "`n(sse exited with $LASTEXITCODE)" -ForegroundColor Yellow }
Remove-Item $sseFile.FullName -ErrorAction SilentlyContinue

Write-Host "`n$sep" -ForegroundColor DarkGray
Write-Host " ALL ENDPOINT STEPS DONE" -ForegroundColor Green
Write-Host $sep -ForegroundColor DarkGray
Write-Host "This conversation was persisted to data/episodic/ (real training data)."
Write-Host "Next: python scripts\export_lora_data.py && python scripts\finetune_lora.py --check-only"
