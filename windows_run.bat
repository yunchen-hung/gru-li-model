@echo off
REM 

REM 
for /f "tokens=2 delims==" %%A in ('wmic cpu get NumberOfLogicalProcessors /value') do set CORES=%%A

REM 
if "%CORES%"=="" set CORES=8

echo Detected %CORES% logical processors.

REM 
set OMP_NUM_THREADS=%CORES%
set MKL_NUM_THREADS=%CORES%
set KMP_AFFINITY=granularity=fine,compact,1,0
set KMP_BLOCKTIME=1

REM 
powercfg /setactive SCHEME_MIN >nul 2>&1

REM 
python main.py %*

echo.
echo Script finished. Press any key to exit.
pause >nul
