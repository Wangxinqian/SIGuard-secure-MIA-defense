# SIGuard_SecureMIA_Defense

Codebase for the paper:

> **SIGuard: Guarding Secure Inference with Post Data Privacy**  
> Accepted to **NDSS 2025**

This repository contains the implementation of **SIGuard**, a defense framework that protects against **membership inference attacks (MIA)** under **secure multi-party computation (MPC)** environments.

---

## 🧩 Repository Contents

| File               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `MemGuard.py`      | MemGuard (CCS 2019) implementation, four softmax approximations used in MPC |
| `SiGuard.mpc`      | SIGuard                                                                     |
| `defender_model.py`| Membership classifier                                                       |
| `ml.py`            | machine learning functions used in MP-SPDZ                                  |

---


### Step 1: Install MP-SPDZ

Follow the instructions in the [MP-SPDZ repository](https://github.com/data61/MP-SPDZ):

```bash
git clone https://github.com/data61/MP-SPDZ.git
cd MP-SPDZ
make
```

### Step 2: Place the Files

Move the following files into your MP-SPDZ project directory:

```bash
# MPC protocol file
cp SiGuard.mpc MP-SPDZ/Programs/Source/

# Training and evaluation logic
cp defender_model.py ml.py MP-SPDZ/Compiler/
```

### Step 3: Run Secure Evaluation

Use the standard MP-SPDZ workflow to compile and run your protocol. For example:

```bash
cd MP-SPDZ
Scripts/setup-ssl.sh 3
make -j 8 replicated-ring-party.x
./compile.py -M -R 64 SiGuard
Scripts/ring.sh SiGuard
```

## 📜 Citation

If you use this codebase in your research, please cite:

```bibtex
@inproceedings{wang2025siguard,
  author={Xinqian Wang and Xiaoning Liu and Shangqi Lai and Xun Yi and Xingliang Yuan},
  title={SIGuard: Guarding Secure Inference with Post Data Privacy},
  booktitle={NDSS},
  year={2025}
}
