# Assignment 1

This folder contains the written answer, source data, generated figures, and
Python code for Assignment 1.

## Files

- `Assignment 1.pdf`: original assignment prompt.
- `Ans.md`: written responses in Markdown.
- `problem1.csv` through `problem5.csv`: input data.
- `Codes/ass1.py` through `Codes/ass5.py`: code used to generate the reported
  numbers and plots.
- `Codes/requirement.txt`: Python package requirements.
- `Codes/run_all.sh`: one-command runner that executes all scripts and writes
  the combined output to `Codes/results.txt`.
- `Codes/export_pdf.sh`: converts `Ans.md` to `Ans.pdf` with math rendering.
- `Codes/results.txt`: combined output from the most recent run of
  `Codes/run_all.sh`.
- `PIC/`: generated plots referenced by `Ans.md`.

## Requirements

Use Python with these packages installed:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `matplotlib`

On this machine, the scripts were verified with `/opt/anaconda3/bin/python3`.
The runner will use `/opt/anaconda3/bin/python3` when it exists; otherwise it
falls back to `python3`. To force another interpreter, set `PYTHON_BIN`.

To install the packages into your active Python environment:

```bash
python -m pip install -r Codes/requirement.txt
```

## Reproduce Results

From this `Assignment1` folder, run:

```bash
./Codes/run_all.sh
```

That command runs every script and writes the combined terminal output to
`Codes/results.txt`.

## Generate PDF

This repository has a local Markdown-to-PDF environment at `.venv-mdpdf` using
`md-to-pdf-cli`. It renders LaTeX math with KaTeX and embeds the local plot
images.

From this `Assignment1` folder, run:

```bash
./Codes/export_pdf.sh
```

The output is `Ans.pdf`. To choose a different input or output file:

```bash
./Codes/export_pdf.sh Ans.md Ans.pdf
```

If the executable bit is not preserved after download, use
`bash Codes/run_all.sh` instead.

To choose an explicit Python interpreter:

```bash
PYTHON_BIN=python ./Codes/run_all.sh
```

To write the combined output to a different file:

```bash
./Codes/run_all.sh Codes/my_results.txt
```

You can also run the scripts individually:

```bash
/opt/anaconda3/bin/python3 Codes/ass1.py
/opt/anaconda3/bin/python3 Codes/ass2.py
/opt/anaconda3/bin/python3 Codes/ass3.py
/opt/anaconda3/bin/python3 Codes/ass4.py
/opt/anaconda3/bin/python3 Codes/ass5.py
```

If your `python` command already points to an environment with the required
packages, these commands are equivalent:

```bash
python Codes/ass1.py
python Codes/ass2.py
python Codes/ass3.py
python Codes/ass4.py
python Codes/ass5.py
```

The scripts use paths relative to their own files, so they can also be run from the repository root.
