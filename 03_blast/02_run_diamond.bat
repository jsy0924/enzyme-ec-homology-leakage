@echo off
cd /d "%~dp0"

echo ============================================================
echo  STEP 2 (optional) : all-vs-all DIAMOND blastp
echo ============================================================
echo.
echo  hits.tsv is ALREADY INCLUDED in this folder, so you do not
echo  need to run this. It is here so the search can be repeated.
echo.
echo  This step needs diamond.exe (version 2.1.10 was used) placed
echo  in this folder, and all.fasta made by 01_make_fasta.py.
echo.

if not exist diamond.exe goto no_diamond
if not exist all.fasta goto no_fasta

diamond.exe makedb --in all.fasta -d all
if errorlevel 1 goto failed

diamond.exe blastp -d all -q all.fasta -o hits.tsv --very-sensitive --evalue 1e-3 --max-target-seqs 0 --outfmt 6 qseqid sseqid pident length evalue bitscore qcovhsp scovhsp
if errorlevel 1 goto failed

echo.
echo  DONE. hits.tsv written.
goto end

:no_diamond
echo  diamond.exe not found in this folder. Nothing was changed.
echo  The included hits.tsv can be used as it is.
goto end

:no_fasta
echo  all.fasta not found. Run 01_make_fasta.py first.
goto end

:failed
echo  DIAMOND reported an error.

:end
echo.
pause
