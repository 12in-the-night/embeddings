@echo off
REM ============================================================
REM  FULL PIPELINE - setup, verify, smoke, train, compress
REM ============================================================
REM
REM  USAGE
REM  -----
REM    pipeline.bat                 Full run: setup, verify, smoke, train, compress
REM    pipeline.bat skip_setup      Skip venv + pip install
REM    pipeline.bat skip_smoke      Skip the 20-step smoke test
REM    pipeline.bat train_only      Only run training (no setup, smoke, compress)
REM    pipeline.bat compress_only   Only run compression (assumes training done)
REM
REM  BEFORE RUNNING
REM  --------------
REM    1. Set HF_TOKEN below if using gated datasets (LAION)
REM    2. Ensure config.yaml and requirements.txt are in the same folder
REM
REM  OUTPUT
REM  ------
REM    ckpts_pretrain/         training checkpoints
REM    ckpts_compressed/       pruned + quantized models
REM    pipeline.log            full log, appended across stages
REM
REM ============================================================
setlocal EnableDelayedExpansion

set ARGS=%*
set MODE=%~1

REM ------------------------------------------------------------
REM  Environment
REM ------------------------------------------------------------
set XLA_PYTHON_CLIENT_PREALLOCATE=false
set XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

REM Optional: set HF token here or via system env
REM set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx

set CKPT_DIR=ckpts_pretrain
set COMPRESS_DIR=ckpts_compressed
set LOG_FILE=pipeline.log

echo ============================================================
echo  Multimodal MoE Embedding - Full Pipeline
echo  Mode: %MODE%   Args: %ARGS%
echo ============================================================
echo.

REM ============================================================
REM  [1/6] SETUP
REM ============================================================
if /I "%MODE%"=="skip_setup"    goto :skip_setup
if /I "%MODE%"=="train_only"    goto :skip_setup
if /I "%MODE%"=="compress_only" goto :skip_setup

echo [1/6] SETUP
if not exist venv (
    echo    creating venv ...
    python -m venv venv
) else (
    echo    venv exists, skipping creation
)
call venv\Scripts\activate.bat

echo    upgrading pip ...
python -m pip install --upgrade pip > nul 2>&1

echo    installing requirements ...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed
    goto :fail
)

:skip_setup
if not exist venv (
    echo ERROR: venv not found. Run without skip_setup first.
    goto :fail
)
call venv\Scripts\activate.bat

REM ============================================================
REM  [2/6] VERIFY
REM ============================================================
echo.
echo [2/6] VERIFY
python -c "import jax; d=jax.devices(); print('    devices:', d); print('    count:', len(d)); print('    platform:', d[0].platform if d else 'none')"
if errorlevel 1 (
    echo ERROR: JAX device check failed
    goto :fail
)

if not exist config.yaml (
    echo ERROR: config.yaml not found in current directory
    goto :fail
)
echo    config.yaml found

REM ============================================================
REM  [3/6] SMOKE TEST  (allnli only - no LAION download)
REM ============================================================
echo %ARGS% | findstr /I "skip_smoke" > nul
if not errorlevel 1 goto :skip_smoke
if /I "%MODE%"=="train_only"    goto :skip_smoke
if /I "%MODE%"=="compress_only" goto :skip_smoke

echo.
echo [3/6] SMOKE TEST
echo    generating config_smoke.yaml ...
python -c "import yaml; c=yaml.safe_load(open('config.yaml')); c['training'].update(steps=20,log_every=1,batch=8,seq_len=64,precision='fp32'); c['model'].update(d_model=128,n_layers=4); c['moe'].update(n_experts=4); c['image'].update(size=64,dim=64,layers=2,heads=4); c['data_mix'].update(image_text=0,msmarco_triplet=0,bekko_triplet=0,allnli_triplet=1); yaml.dump(c,open('config_smoke.yaml','w'))"

set MM_CONFIG=config_smoke.yaml
echo    running 20-step smoke test ...
python main.py >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo ERROR: smoke test failed. Check %LOG_FILE%
    goto :fail
)
echo    smoke test passed
set MM_CONFIG=
del config_smoke.yaml > nul 2>&1

:skip_smoke

REM ============================================================
REM  [4/6] TRAIN
REM ============================================================
if /I "%MODE%"=="compress_only" goto :skip_train

echo.
echo [4/6] TRAIN
echo    starting training (this will take a while)
echo    log: %LOG_FILE%
if not exist %CKPT_DIR% mkdir %CKPT_DIR%

set RESUME_ARG=
if exist %CKPT_DIR%\pretrain_final.pkl (
    echo    found existing checkpoint, resuming
    set RESUME_ARG=--resume %CKPT_DIR%\pretrain_final.pkl
)

python main.py !RESUME_ARG! 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOG_FILE%' -Append"
if errorlevel 1 (
    echo ERROR: training failed. Check %LOG_FILE%
    goto :fail
)
echo    training complete

:skip_train

REM ============================================================
REM  [5/6] COMPRESS
REM ============================================================
echo.
echo [5/6] COMPRESS
if not exist %COMPRESS_DIR% mkdir %COMPRESS_DIR%

set SRC=%CKPT_DIR%\pretrain_final.pkl
if not exist "%SRC%" (
    echo    no trained checkpoint found, skipping compression
    goto :skip_compress
)

echo    step 1: prune experts ...
python main.py --action prune_experts --input "%SRC%" --output %COMPRESS_DIR%\pruned_experts.pkl >> %LOG_FILE% 2>&1
if errorlevel 1 goto :fail

echo    step 2: prune layers ...
python main.py --action prune_layers --input %COMPRESS_DIR%\pruned_experts.pkl --output %COMPRESS_DIR%\pruned_layers.pkl >> %LOG_FILE% 2>&1
if errorlevel 1 goto :fail

echo    step 3: AWQ calibration ...
python main.py --action awq --input %COMPRESS_DIR%\pruned_layers.pkl --output %COMPRESS_DIR%\awq.pkl >> %LOG_FILE% 2>&1
if errorlevel 1 goto :fail

echo    step 4: export quantized bundle ...
python main.py --action export --input %COMPRESS_DIR%\awq.pkl --output %COMPRESS_DIR%\int_bundle.pkl >> %LOG_FILE% 2>&1
if errorlevel 1 goto :fail

echo    compression complete

:skip_compress

REM ============================================================
REM  [6/6] SUMMARY
REM ============================================================
echo.
echo [6/6] SUMMARY
echo.
echo    --- checkpoints ---
if exist %CKPT_DIR% (
    for %%F in (%CKPT_DIR%\*.pkl) do (
        for %%S in ("%%F") do echo    %%~nxS  %%~zS bytes
    )
)
echo.
echo    --- compressed ---
if exist %COMPRESS_DIR% (
    for %%F in (%COMPRESS_DIR%\*.pkl) do (
        for %%S in ("%%F") do echo    %%~nxS  %%~zS bytes
    )
)
echo.
echo    --- log ---
if exist %LOG_FILE% (
    for %%F in ("%LOG_FILE%") do echo    %%~nxF  %%~zF bytes
)

echo.
echo ============================================================
echo  PIPELINE COMPLETE
echo ============================================================
goto :end

:fail
echo.
echo ============================================================
echo  PIPELINE FAILED - check %LOG_FILE%
echo ============================================================
endlocal
exit /b 1

:end
endlocal
exit /b 0
