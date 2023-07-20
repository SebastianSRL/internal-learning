# Seismic Source Recovery Algorithm via Internal Learning in the Cross-spread Domain: A Tensorflow Implementation
This is a Tensorflow implementation of the proposed work in ["Seismic Source Recovery Algorithm via Internal Learning in the Cross-spread Domain"]([https://ieeexplore.ieee.org/abstract/document/9592023](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=TWFD08sAAAAJ&citation_for_view=TWFD08sAAAAJ:qjMakFHDy7sC)) (Sebastián Rivera, Iván Ortíz, Tatiana Gelvez, Laura Galvis, Henry Arguello, EAGE, 2022).

![Imagen1](https://github.com/SebastianSRL/internal-learning/assets/66753336/3c26a1a3-7af2-4dda-8a83-89145474cca7)

## Project structure 
```bash
.
├── ...
├── data                   # Folder to place data to reconstruct.
│   └── cube.npy            # Tridimensional array (HxWxC).
├── src                    # Source code.
│   ├── config.yml          # Hyperparameters to change.
│   ├── default.py          # Default hyperparameters.
│   ├── metrics.py          # Metrics to measure the performance.
│   ├── models.py           # Neural network architecture.
│   ├── preprocessing.py    # Preprocessing operations before the internal learning.
│   ├── utils.py            # Utils functions.
│   └── main.py             # Internal learning training.
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
## Usage 

To avoid Tensorflow and cuda compatibility issues we employ and recommend Docker. 
After install Docker execute the following command in the project root folder: 

```bash
docker-compose up 
```
## Results
The following image shows the comparison between the reconstructed shots obtained via the proposed method and other reconstruction methods.

![Imagen2](https://github.com/SebastianSRL/internal-learning/assets/66753336/6c6e2ccd-021d-48f8-b809-20bff3f1d11f)

## Cite

```
@inproceedings{rivera2022seismic,
  title={Seismic Source Recovery Algorithm via Internal Learning in the Cross-spread Domain},
  author={Rivera, S and Ortiz, I and Gelvez-Barrera, T and Galvis, L and Arguello, H},
  booktitle={Fourth HGS/EAGE Conference on Latin America},
  volume={2022},
  number={1},
  pages={1--5},
  year={2022},
  organization={European Association of Geoscientists \& Engineers}
}
```
## License 
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
