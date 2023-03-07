

# 3D Geometry Deep Learning in Drugs
![contributing-image](figures/3D-plus.png)

related to  **Geometric Deep Learning** and **3D** for  **Drugs**.

**Updating ...**  


## Menu

- [3D Geometry Deep Learning in Drugs](#3d-geometry-deep-learning-in-drugs)
  - [Menu](#menu)
  - [Reviews](#reviews)
  - [Datasets and Benchmarks](#datasets-and-benchmarks)
    - [Datasets](#datasets)
    - [Benchmarks](#benchmarks)
  - [Small molecular application](#small-molecular-application)
    - [Property prediction](#property-prediction)
    - [Molecular conformation generation](#molecular-conformation-generation)
    - [Molecular generation](#molecular-generation)
  - [Macro-molecular application](#macro-molecular-application)
    - [Property prediction](#property-prediction-1)
    - [Binding sites prediction](#binding-sites-prediction)
    - [Bindding score prediction](#bindding-score-prediction)
    - [Bindding conformation prediction](#bindding-conformation-prediction)
    - [Structure-based drug design](#structure-based-drug-design)


## Reviews

* **Structure-based drug design with geometric deep learning**[2023]  
[[Paper]](https://doi.org/10.1016/j.sbi.2023.102548)

* **MolGenSurvey: A Systematic Survey in Machine Learning Models for Molecule Design**[2022]  
[[Paper]](https://arxiv.org/abs/2203.14500)

* **Geometrically Equivariant Graph Neural Networks: A Survey**[2022]  
[[Paper]](https://arxiv.org/abs/2202.07230)

* **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges**[2021]  
[[Paper]](https://arxiv.org/abs/2104.13478)

* **Geometric deep learning on molecular representations**[2021]  
[[Paper]](https://arxiv.org/abs/2107.12375)


## Datasets and Benchmarks

### Datasets

**QM dataset**

http://quantum-machine.org/datasets/

**Qmugs**

https://www.nature.com/articles/s41597-022-01390-7

**GEOM**

https://www.nature.com/articles/s41597-022-01288-4


**OC20**

https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md

**PDBBIND**

http://pdbbind.org.cn/ 

**DIPS**

https://github.com/drorlab/DIPS

### Benchmarks

## Small molecular application
### Property prediction
### Molecular conformation generation
* **GeoDiff:A Geometric Diffusion Model for Molecular Conformation Generation** [2022]   
Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, Jian Tang.  
[Paper](https://arxiv.org/abs/2203.02923) | [code](https://github.com/MinkaiXu/GeoDiff)    

* **Learning Neural Generative Dynamics for Molecular Conformation Generation** [2021]   
Minkai Xu, Shitong Luo, Yoshua Bengio, Jian Peng, Jian Tang.  
[Paper](https://arxiv.org/abs/2102.10240) | [code](https://github.com/MinkaiXu/CGCF-ConfGen)    

* **An End-to-End Framework for Molecular Conformation Generation via Bilevel Programming** [2021]   
Minkai Xu, Wujie Wang, Shitong Luo, Chence Shi, Yoshua Bengio, Rafael Gomez-Bombarelli, Jian Tang.  
[Paper](https://arxiv.org/abs/2105.07246) | [code](https://github.com/MinkaiXu/ConfVAE-ICML21)    

* **A Generative Model for Molecular Distance Geometry** [2020]   
Gregor N. C. Simm, José Miguel Hernández-Lobato.  
[Paper](https://arxiv.org/abs/1909.11459) | [code](https://github.com/gncs/graphdg)    

* **Learning Gradient Fields for Molecular Conformation Generation** [2021]   
Chence Shi, Shitong Luo, Minkai Xu, Jian Tang.  
[Paper](https://arxiv.org/abs/2105.03902) | [code](https://github.com/DeepGraphLearning/ConfGF)    

* **GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles** [2021]   
Octavian-Eugen Ganea, Lagnajit Pattanaik, Connor W. Coley, Regina Barzilay, Klavs F. Jensen, William H. Green, Tommi S. Jaakkola.  
[Paper](https://arxiv.org/abs/2106.07802) | [code](https://github.com/PattanaikL/GeoMol)    

* **Predicting Molecular Conformation via Dynamic Graph Score Matching** [2021]   
Shitong Luo, Chence Shi, Minkai Xu, Jian Tang.  
[Paper](https://www.nature.com/articles/s41592-022-01490-7)

* **Direct Molecular Conformation Generation** [2022]   
Jinhua Zhu, Yingce Xia, Chang Liu, Lijun Wu, Shufang Xie, Yusong Wang, Tong Wang, Tao Qin, Wengang Zhou, Houqiang Li, Haiguang Liu, Tie-Yan Liu.  
[Paper](https://arxiv.org/abs/2202.01356) | [code](https://github.com/DirectMolecularConfGen/DMCG)    

* **Molecular Geometry Prediction using a Deep Generative Graph Neural Network** [2019]   
Elman Mansimov, Omar Mahmood, Seokho Kang, Kyunghyun Cho.  
[Paper](https://arxiv.org/abs/1904.00314) | [code](https://github.com/nyu-dl/dl4chem-geometry)    


### Molecular generation
* **Equivariant Diffusion for Molecule Generation in 3D** [2022]   
Emiel Hoogeboom, Victor Garcia Satorras, Clément Vignac, Max Welling.  
[Paper](https://arxiv.org/abs/2203.17003) | [code](https://github.com/ehoogeboom/e3_diffusion_for_molecules)    

* **E(n) Equivariant Normalizing Flows** [2021]   
Victor Garcia Satorras, Emiel Hoogeboom, Fabian Fuchs, Ingmar Posner, Max Welling.  
[Paper](https://arxiv.org/abs/2105.09016) | [code](https://github.com/vgsatorras/en_flows)    

* **Symmetry-Aware Actor-Critic for 3D Molecular Design** [2021]   
Gregor N. C. Simm, Robert Pinsler, Gábor Csányi, José Miguel Hernández-Lobato.  
[Paper](https://arxiv.org/abs/2011.12747) | [code](https://github.com/gncs/molgym)    

* **Reinforcement Learning for Molecular Design Guided by Quantum Mechanics** [2020]   
Gregor N. C. Simm, Robert Pinsler, José Miguel Hernández-Lobato.  
[Paper](https://arxiv.org/abs/2002.07717) | [code](https://github.com/gncs/molgym)    

* **An Autoregressive Flow Model for 3D Molecular Geometry Generation from Scratch** [2022]   
Youzhi Luo, Shuiwang Ji.  
[Paper](https://iclr.cc/virtual/2022/poster/7066) | [code](https://github.com/divelab/DIG)    

* **Inverse design of 3d molecular structures with conditional generative neural networks** [2021]   
Niklas W. A. Gebauer, Michael Gastegger, Stefaan S. P. Hessmann, Klaus-Robert Müller, Kristof T. Schütt.  
[Paper](https://arxiv.org/abs/2109.04824) | [code](http://www.github.com/atomistic-machine-learning/cG-SchNet)    

* **3D-Scaffold: A Deep Learning Framework to Generate 3D Coordinates of Drug-like Molecules with Desired Scaffolds** [2021]   
Rajendra P Joshi, Niklas W A Gebauer, Mridula Bontha, Mercedeh Khazaieli, Rhema M James, James B Brown, Neeraj Kumar.  
[Paper](https://pubmed.ncbi.nlm.nih.gov/34662142/) | [code](https://github.com/PNNL-CompBio/3D_Scaffold)    

* **Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules** [2019]   
Niklas W. A. Gebauer, Michael Gastegger, Kristof T.  
[Paper](https://arxiv.org/abs/1906.00957) | [code](www.github.com/atomistic-machine-learning/G-SchNet)    


## Macro-molecular application
### Property prediction
### Binding sites prediction  
* **ScanNet: an interpretable geometric deeplearning model for structure-based proteinbinding site prediction** [2022]   
Tubiana, Jérôme, Schneidman-Duhovny, Dina, Wolfson, Haim J.  
[Paper](https://www.nature.com/articles/s41592-022-01490-7) | [code](https://github.com/jertubiana/ScanNet)    

* **Geometric Transformers for Protein Interface Contact Prediction** [2022]  
Alex Morehead, Chen Chen, Jianlin Cheng.  
[Paper](https://arxiv.org/abs/2110.02423) | [code](https://github.com/BioinfoMachineLearning/DeepInteract)  

* **Fast end-to-end learning on protein surfaces** [2021]  
Freyr Sverrisson, Jean Feydy, Bruno E Correia, Michael M Bronstein.     
[Paper](https://ieeexplore.ieee.org/document/9577686) | [code](https://github.com/FreyrS/dMaSIF)  

* **Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning** [2020]  
Gainza et al.  
[Paper](https://www.nature.com/articles/s41592-019-0666-6) | [code](https://github.com/lpdi-epfl/masif)  

* **DeepSite: protein-binding site predictor using 3D-convolutional neural networks** [2017]  
Jiménez et al.  
[Paper](https://pubmed.ncbi.nlm.nih.gov/28575181/)

### Bindding score prediction
* **Geometric Interaction Graph Neural Network for Predicting Protein-Ligand Binding Affinities from 3D Structures (GIGN).** [2023]   
Ziduo Yang, Weihe Zhong, Qiujie Lv, Tiejun Dong, Calvin Yu-Chian Chen.  
[Paper](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03906) | [code](https://github.com/guaguabujianle/GIGN)    

* **Predicting Drug-Target Interaction Using a Novel Graph Neural Network with 3D Structure-Embedded Graph Representation** [2019]   
Jaechang Lim, Seongok Ryu, Kyubyong Park, Yo Joong Choe, Jiyeon Ham, Woo Youn Kim.  
[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00387) | [code](https://github.com/jaechanglim/GNN_DTI)    

* **Graph Convolutional Neural Networks for Predicting Drug-Target Interactions** [2019]   
Wen Torng, Russ B. Altman.  
[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00628)   

* **PIGNet: a physics-informed deep learning model toward generalized drug–target interaction predictions** [2020]   
Seokhyun Moon, Wonho Zhung, Soojung Yang, Jaechang Lim, Woo Youn Kim.  
[Paper](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d1sc06946b#!) | [code](https://github.com/ACE-KAIST/PIGNet)    

* **Multi-Scale Representation Learning on Proteins** [2022]   
Vignesh Ram Somnath, Charlotte Bunne, Andreas Krause.  
[Paper](https://arxiv.org/abs/2204.02337) | [code](https://github.com/vsomnath/holoprot)    

* **InteractionGraphNet: A Novel and Efficient Deep Graph Representation Learning Framework for Accurate Protein-Ligand Interaction Predictions** [2021]   
Dejun Jiang, Chang-Yu Hsieh, Zhenxing Wu, Yu Kang, Jike Wang, Ercheng Wang, Ben Liao, Chao Shen, Lei Xu, Jian Wu, Dongsheng Cao, Tingjun Hou.  
[Paper](https://pubs.acs.org/doi/10.1021/acs.jmedchem.1c01830) | [code](https://github.com/zjujdj/InteractionGraphNet/tree/master)    

* **Structure-aware Interactive Graph Neural Networks for the Prediction of Protein-Ligand Binding Affinity** [2021]   
Shuangli Li, Jingbo Zhou, Tong Xu, Liang Huang, Fan Wang, Haoyi Xiong, Weili Huang, Dejing Dou, Hui Xiong.  
[Paper](https://arxiv.org/abs/2107.10670)  

* **AtomNet: A Deep Convolutional Neural Network for Bioactivity Prediction in Structure-based Drug Discovery** [2021]   
Izhar Wallach, Michael Dzamba, Abraham Heifets.  
[Paper](https://arxiv.org/abs/1510.02855)

* **KDEEP: Protein-Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks** [2018]   
José Jiménez et al.  
[Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00650)

### Bindding conformation prediction
* **DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking** [2022]   
Gabriele Corso, Hannes Stärk, Bowen Jing, Regina Barzilay, Tommi S Jaakkola.  
[Paper](https://doi.org/10.48550/arXiv.2210.01776)  

* **EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction** [2022]   
Hannes Stärk, Octavian-Eugen Ganea, Lagnajit Pattanaik, Regina Barzilay, T Jaakkola.  
[Paper](https://arxiv.org/abs/2202.05146) | [code](https://github.com/HannesStark/EquiBind)  

* **A geometric deep learning approach to predict binding conformations of bioactive molecules** [2021]   
Arne Schneuing et al.  
[Paper](https://www.nature.com/articles/s42256-021-00409-9) | [code](https://github.com/OptiMaL-PSE-Lab/DeepDock)  

* **EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction** [2021]   
Hannes Stärk, Octavian-Eugen Ganea, Lagnajit Pattanaik, Regina Barzilay, Tommi Jaakkola.  
[Paper](https://arxiv.org/abs/2202.05146) | [code](https://github.com/HannesStark/EquiBind)  

* **DeepBSP—a machine learning method for accurate prediction of protein–ligand docking structures** [2021]   
Jingxiao Bao, Xiao He,* and John Z. H. Zhang*.  
[Paper](https://pubmed.ncbi.nlm.nih.gov/33979150/) | [code](https://github.com/BaoJingxiao/DeepBSP)  

### Structure-based drug design
* **Structure-based Drug Design with Equivariant Diffusion Models** [2022]   
Arne Schneuing et al.  
[Paper](https://doi.org/10.48550/arXiv.2210.13695) | [code](https://github.com/arneschneuing/DiffSBDD)  

* **Equivariant 3D-Conditional Diffusion Models for Molecular Linker Design** [2022]   
Ilia Igashov et al.  
[Paper](https://arxiv.org/abs/2210.05274) | [code](https://github.com/igashov/DiffLinker)  

* **Fragment-Based Ligand Generation Guided By Geometric Deep Learning On Protein-Ligand Structure** [2022]   
Alexander Powers, Helen Yu, Patricia Suriana, Ron Dror.  
[Paper](https://www.semanticscholar.org/paper/Fragment-Based-Ligand-Generation-Guided-By-Deep-On-Powers-Yu/8405e5a5491d871a01c6a6ec2507a5151b437560) | [code](https://github.com/marcopodda/fragment-based-dgm)  

* **Generating 3D Molecules for Target Protein Binding** [2022]   
Meng Liu, Youzhi Luo, Kanji Uchino, Koji Maruhashi, Shuiwang Ji.  
[Paper](https://arxiv.org/abs/2204.09410) | [code](https://github.com/divelab/GraphBP)  

* **3DLinker: An E(3) Equivariant Variational Autoencoder for Molecular Linker Design** [2022]   
Yinan Huang, Xingang Peng, Jianzhu Ma, Muhan Zhang.  
[Paper](https://proceedings.mlr.press/v162/huang22g.html) | [code](https://github.com/YinanHuang/3DLinker)  

* **Structure-based de novo drug design using 3D deep generative models** [2021]   
Yibo Li, Jianfeng Pei, Luhua Lai.  
[Paper](https://pubmed.ncbi.nlm.nih.gov/34760151/)  

* **A 3D generative model for structure-based drug design** [2021]   
Shitong Luo, Jiaqi Guan, Jianzhu Ma, Jian Peng  
[Paper](https://arxiv.org/abs/2203.10446) | [code](https://github.com/luost26/3D-Generative-SBDD)  

* **Generating 3d molecular structures conditional on a receptor binding site with deep generative models** [2020]   
Tomohide Masuda, Matthew Ragoza, David Ryan Koes.  
[Paper](https://arxiv.org/abs/2010.14442) | [code](https://github.com/mattragoza/LiGAN)   