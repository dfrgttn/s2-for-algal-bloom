# Description
Implementation and data of publication: Grendaitė, Dalia, and Linas Petkevičius. "[Identification of algal blooms in lakes in the Baltic states using Sentinel-2 data and artificial neural networks](https://ieeexplore.ieee.org/abstract/document/10436679)." IEEE Access 12 (2024): 27973-27988.

This repository holds material of a project for algal bloom identification from optical satellite data (Sentinel-2 MSI) based on data from lakes and ponds in the Baltic states.

# Data
The annotated dataset is available in the [link](https://drive.google.com/file/d/1BxOFQCDSEcenDH_TlsZnXGNhLIZLZ-c5/view?usp=sharing). 

# Code
A code in the file [link](https://github.com/dfrgttn/s2-for-algal-bloom/blob/main/Code_for_data_retrieval_from_GEE.ipynb) decribes how to retrieve Sentinel-2 data for one lake and for many lakes. An example is based on downloading data for 357 lakes and ponds in Lithuania. The code holds names of the measurement sites, reflectance in Sentinel-2 bands and additional radiometric indices that can be used for prediction of algal blooms. The column "bloom" defines whether the lake is blooming (1) or not (0).

## Funding information
This project was funded by the European Union (project No S-MIP-23-44) under the agreement with the Research Council of Lithuania (LMTLT).

## Cite:

  Grendaitė, Dalia, and Linas Petkevičius. "Identification of algal blooms in lakes in the Baltic states using Sentinel-2 data and artificial neural networks." IEEE Access 12 (2024): 27973-27988.

or

    @article{grendaite2024identification,
      title={Identification of algal blooms in lakes in the Baltic states using Sentinel-2 data and artificial neural networks},
      author={Grendait{\.e}, Dalia and Petkevi{\v{c}}ius, Linas},
      journal={IEEE Access},
      volume={12},
      pages={27973--27988},
      year={2024},
      publisher={IEEE}
    }
