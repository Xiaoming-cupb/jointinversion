# JointINV

JointINV is a deep-learning workflow for seismic impedance inversion using **well-log constraints**, **forward-model consistency**, and **frequency-domain regularization**.  
It provides a complete pipeline from **well–seismic preprocessing** to **2D line construction**, **model training**, and **full-volume prediction**.

---

## Repository Structure

```text
JointINV/
├── build2Dlines/        # Build 2D training/validation samples from 3D volumes
├── preprocessdata/      # Preprocess seismic, model, and well data
├── train_predict/       # Network training and full-volume prediction
└── README.md

A recommended detailed structure is:

JointINV/
├── preprocessdata/
│   └── pre_seis_and_wells.py
├── build2Dlines/
│   └── train_data2d.py
├── train_predict/
│   ├── train.py
│   ├── train_config.yaml
│   ├── predict_with_denorm_from_wells.py
│   ├── predict.yaml
│   ├── net_torch.py
│   ├── imploss.py
│   └── torchfilters.py
└── README.md

**Workflow**

The workflow contains three main stages:

**1. Preprocess data**

Folder: preprocessdata/

This step is used to:

read 3D seismic data from SEGY
export seismic volume to binary .dat
export model volume such as RGT if needed
save metadata to metadata.json
read raw well files
map well coordinates to seismic inline/xline
resample well logs onto the seismic time grid
export processed well files for later training and prediction

Typical outputs:

out/
├── metadata.json
├── seis/
│   ├── seis.dat
│   └── model.dat
└── well/
    ├── Well_1.txt
    ├── Well_2.txt
    └── ...

**2. Build 2D lines**

Folder: build2Dlines/

This step generates 2D samples from 3D seismic and impedance volumes.

Generated data usually include:

sx: seismic sections
ws: smoothed impedance sections
wx: sparse well-log sections

Typical output structure:

out/seis/train/
├── trainimp/
│   ├── sx/
│   ├── ws/
│   └── wx/
└── validimp/
    ├── sx/
    ├── ws/
    └── wx/

This stage supports:

automatic well scanning
validation split by well
random path-based extraction
single-well and multi-well sample generation
**3. Train and predict**

Folder: train_predict/

This step contains:

model training
validation
checkpoint saving
diagnostic image logging
full-volume prediction
optional denormalization using well statistics

The model uses three types of constraints:

well-log loss
reconstruction loss
spectrum loss

The total loss is:

Loss = alpha0 * Lwell + alpha1 * Lrecons + alpha2 * Lspec

How to Run
Step 1. Preprocess data

cd preprocessdata
python pre_seis_and_wells.py

Step 2. Build 2D training samples

cd ../build2Dlines
python train_data2d.py

Step 3. Train the model

cd ../train_predict
python train.py --config train_config.yaml

Step 4. Predict full impedance volume

cd ../train_predict
python predict_with_denorm_from_wells.py

**Data Format**
Volume data

Binary .dat files stored as float32 with shape:

(n_inline, n_xline, n_time)

**Metadata**

metadata.json typically contains:

n1, d1, f1
n_inline, n_xline
d_inline, d_xline
inline0, xline0
Well files

**Processed well files are stored as:**

iline xline TWT IMP

Dependencies

**Main dependencies include:**

Python 3.9+
NumPy
PyTorch
PyYAML
Matplotlib
tqdm
pandas
scipy
xarray
segysak
tensorboard

Install with:

pip install numpy torch pyyaml matplotlib tqdm pandas scipy xarray segysak tensorboard

**Notes**
Update all paths in the scripts or YAML files before running.
The workflow currently uses binary .dat files for efficient 3D volume storage.
Prediction can optionally denormalize impedance using processed well-log statistics.
The project is designed for 3D seismic data with 2D extracted training samples.
Citation

If you use this repository in your research, please cite the related paper or acknowledge the corresponding project.

@misc{jointinv,
  title = {JointINV: Joint Well-Constrained Seismic Impedance Inversion with Reconstruction and Spectral Regularization},
  author = {Your Name},
  year = {2026},
  note = {GitHub repository}
}

**License**

Choose a license for your repository, for example:

MIT License
Apache-2.0 License

**Contact**

For questions, suggestions, or collaboration, please contact Sunxiaoming0305@ustc.edu.cn
