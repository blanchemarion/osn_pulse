# OSN Pulse Analysis Draft

## Repository layout

```text
├─ Process/
│  ├─ prepare_for_dunl_<x>.py      # prepares data for supervised DUNL
│  └─ prepare_data.py              # process data; saves animals_data_processed.pkl
├─ dunl/                           # DUNL preparation, training, and visualization
│  ├─ config/
│  └─ src/
│     ├─ preprocess_scripts/       # prepare data
│     ├─ postprocess_scripts/      # inference and analysis scripts
│     └─ train_<x>.py              # train
├─ Estimate_Kernels/
│  ├─ glm/                         # GLM design, fit, and analysis
│  └─ scripts/                     # reverse corr, voxel timing, linear-reg deconv, ...
├─ Notebooks/                      # visualize raw data
├─ functions.py                    # utilities
└─ plots.py                        # plotting helpers
```


---

## DUNL workflow 

Original Repository: https://github.com/btolooshams/dunl-compneuro

**Steps**

1. **Prepare inputs / scripts**  
   Modify the config file
   Run:
   ```bash
   python src\preprocess_scripts\preprocess_data_olfactory_<x>.py
   ```
   Run: 
   ```bash
   python src\preprocess_scripts\prepare_data_and_save_<x>.py
   ```

2. **Train DUNL**  
   Run: 
   ```bash
   python train_<x>.py
   ```

2. **Inference**  
   Run: 
   ```bash
   python src\postprocess_scripts\infersave_data_<x>.py
   ```

3. **Plots and Analysis**  
   All the scripts in `src\postprocess_script`

---

## GLM workflow

**1) Build design matrices**  
```bash
python build_design_matrices_glm_<variant>.py
```
**2) Fit GLM**  
```bash
python fit_glm.py
```
**3) Analyze the fit**  
Choose either:
```bash
python analyse_glm_fit.py
```
or open the exploratory notebook:
```bash
jupyter notebook analyse_glm.ipynb
```

