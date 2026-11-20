# Explainable Cross-Modal Appliance Event Classification with a Small Language Model


> **Explainable Cross-Modal Appliance Event Classification with a Small Language Model**  
> **Ubaid Ahmed**¹, **Ahsan Raza Khan**², **Rao Naveed Bin Rais**³, **Sajjad Hussain**¹, and **Ahmed Zoha**¹  
> ¹ *James Watt School of Engineering, University of Glasgow, Glasgow, UK*  
> ² *School of Engineering, University of Hull, Hull, UK*  
> ³ *Artificial Intelligence Research Centre (AIRC), Ajman University, Ajman, UAE*

---

## 📖 Overview

Non-Intrusive Load Monitoring (NILM) models often require large volumes of labelled multi-household data and operate as black boxes. This repository introduces an **Explainable Cross-Modal (XCM)** framework for **One-vs-Rest (OvR)** appliance event classification under limited labelled data and class imbalance (4:1 negative-to-positive ratio).

### Key Highlights
* **Cross-Modal Representation Fusion:** Integrates a LoRA-tuned **Llama-3.2-1B** Small Language Model (SLM) encoder (training only **0.069%** of parameters) with a lightweight **2-layer MLP** (64 hidden units per layer) to jointly process semantic event descriptions and numerical electrical signatures.
* **Multi-Rate Feature Extraction (UK-DALE Household 1):** Detects active appliance events on low-frequency power channels (\(1/6\text{ Hz}\)) and extracts steady-state and transient electrical/harmonic features (\(V_{\text{rms}}\), \(I_{\text{rms}}\), \(P\), \(S\), \(Q\), Power Factor, Crest Factor, and 3rd, 5th, and 7th order FFT harmonics) from \(16\text{ kHz}\) high-frequency waveforms across time offsets \(T \in \{0, 3, 8, 15, 30, 60\}\text{ s}\).
* **Built-in Modality-Ablation Explainability:** Isolates semantic (SLM) vs. numerical (MLP) confidence shifts and identifies dominant physical decision anchors (e.g., peak onset power and harmonic ratios) for every inference event.
* **Strong Performance:** Improves average Matthews Correlation Coefficient (MCC) by **6.38%** over Seq2point and **13.64%** over BERT4NILM across five target appliances (*Kettle, Fridge, Dishwasher, Washing Machine, and Microwave*), achieving **0.086 s** per-event inference latency.

---

## 📂 Repository Structure

```text
Explainable-Cross-Modal-Appliance-Event-Classification/
├── data_collection.ipynb           # Stage 1: Low-frequency event detection & 16 kHz electrical/harmonic feature extraction
├── sequence_data_generator.ipynb   # Stage 2: Hybrid dataset construction (numerical vectors + natural-language event prompts)
├── LLM-MLP.ipynb                   # Stage 3: XCM (LoRA Llama-3.2-1B + MLP) training, OvR evaluation, & modality-ablation XAI
└── README.md
```

---

## ⚙️ Installation & Prerequisites

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd Explainable-Cross-Modal-Appliance-Event-Classification
   ```

2. **Create a Python environment and install dependencies:**
   ```bash
   conda create -n xcm-nilm python=3.10 -y
   conda activate xcm-nilm
   pip install torch transformers peft accelerate numpy pandas scipy scikit-learn matplotlib jupyter
   ```

3. **Dataset & Model Access:**
   * **UK-DALE Dataset:** Download Household 1 low-frequency ($1/6\text{ Hz}$) and high-frequency ($16\text{ kHz}$) files along with calibration metadata ($C, k_i, k_v$).
   * **Pretrained SLM:** Ensure you have Hugging Face access configured (`huggingface-cli login`) to load **`meta-llama/Llama-3.2-1B`**.

---

## 🚀 How to Run the Code

Execute the notebooks sequentially in the following three-stage pipeline:

### Stage 1: Event Detection & Feature Extraction (`data_collection.ipynb`)
* Update the file paths in the notebook to point to your local **UK-DALE Household 1** low-frequency and $16\text{ kHz}$ data directories.
* Run all cells to:
  1. Detect active appliance state transitions using appliance power thresholds $[\tau_{\min}, \tau_{\max}]$ and minimum off-duration $\delta_{\text{off}}$.
  2. Extract baseline current $i_q(n)$ ($5\text{ s}$ pre-onset) and incremental current $i_{\Delta,T}(n)$ across offsets $T = \{0, 3, 8, 15, 30, 60\}\text{ s}$ using 320-sample ($20\text{ ms}$) windows.
  3. Compute RMS voltage/current, active/reactive/apparent power, power factor, crest factor, and 3rd/5th/7th harmonics via 320-point FFT.

### Stage 2: Hybrid Cross-Modal Dataset Generation (`sequence_data_generator.ipynb`)
* Run all cells to convert the extracted numerical feature vectors into structured **natural-language temporal descriptions** paired with their standardized numerical feature arrays (30% train / 30% validation / 40% test split across 2,220 valid appliance events).

### Stage 3: XCM Training, Evaluation & Explainability (`LLM-MLP.ipynb`)
* Configure the target appliance for One-vs-Rest (OvR) binary classification (`Kettle`, `Fridge`, `Dishwasher`, `Washing Machine`, or `Microwave`).
* Run all cells to:
  1. Fine-tune the **Llama-3.2-1B** query/key attention projections with LoRA ($r = 8$, $\alpha = 32$, $\text{lr} = 5 \times 10^{-5}$) jointly with the 2-layer MLP ($\text{lr} = 10^{-2}$) for up to 10 epochs with early stopping.
  2. Evaluate Accuracy, MCC, Macro Precision/Recall/F1, ROC-AUC, and confusion matrices.
  3. Generate **modality-ablation explainability traces** comparing semantic vs. numeric probability contributions for individual test events.

---

## 📊 Key Results (UK-DALE Household 1)

| Appliance | Accuracy | MCC | Macro F1 | AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Kettle** | 0.88 | 0.67 | 0.83 | 0.92 |
| **Microwave** | 0.93 | 0.79 | 0.90 | 0.96 |
| **Fridge** | 0.72 | 0.40 | 0.65 | 0.83 |
| **Washing Machine** | 0.78 | 0.34 | 0.67 | 0.71 |
| **Dishwasher** | 0.76 | 0.30 | 0.65 | 0.75 |
| **Average (XCM)** | **0.81** | **0.50** | **0.74** | **0.83** |

---

## 📝 Citation

If you find this code or paper useful in your research, please cite our work:

```bibtex
@inproceedings{ahmed2026explainable,
  title     = {Explainable Cross-Modal Appliance Event Classification with a Small Language Model},
  author    = {Ahmed, Ubaid and Khan, Ahsan Raza and Rais, Rao Naveed Bin and Hussain, Sajjad and Zoha, Ahmed},
  booktitle = {2026 IEEE PES Australasian Universities Power Engineering Conference (AUPEC), https://eprints.gla.ac.uk/396375/},
  year      = {2026},
  note      = {Best Paper Award}
}
```

---

## 🙏 Acknowledgement

This work is supported by the **Deanship of Research and Graduate Studies (DRG), Ajman University, UAE** under Grant ID: `2025-IRG-CEIT-36`.
