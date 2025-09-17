# wrf_python_assimilation

A clean, reproducible repository to run WRF cross-section **data assimilation** experiments with **LETKF**.  
This setup uses **Python** for I/O, obs operators, orchestration, and plotting; and calls your **Fortran LETKF** (via `f2py`) for the analysis step. You can flip between Fortran and a small Python fallback using a config flag.

1) **Create the enviroment**
cd wrf_python_assimilation
conda env create -f environment.yml
conda activate wrf_python_assimilation
2) **Build the Fortran LETKF module**
bash scripts/build_fortran.sh
3) **Prepare cross-sections**
python io/cross_sections.py --config configs/build_cross_sections.yaml
4) **Run an experiment**
bash scripts/run_experiments.sh single configs/single_obs.yaml
bash scripts/run_experiments.sh 2d     configs/full2d_multicycle.yaml
5) **Plot quicklook**
Open utils/Quicklook.ipynb in Jupyter.