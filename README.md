# *lcreg* - Efficient registration of large 3D images 

Rigid and affine registration of large scalar 3D images is an import step for both medical and non-medical image processing. The distinguishing feature of *lcreg* is its capability to efficiently register images that do not fit into system memory. *lcreg* is based on the optimisation of the local correlation similarity measure [1] using a novel image encoding scheme fostering on-the-fly image compression and decompression [2].

# Tutorial and samples
The *lcreg tutorial* provides a step by step guide for the installation and practical application of the software and is complemented by sample data and configuration files (156 MB). These ressources can be downloaded from [here](https://cloud.hs-augsburg.de/index.php/s/iR8BBZM2n6zcxSp).

# Contact and support
ResearchGate members please use the [project page](https://www.researchgate.net/project/Efficient-registration-of-large-3D-images-lcreg) to post comments or ask questions. The email address of the project is lcreg@hs-augsburg.de. 

# Acknowledgements
Many thanks to Karl-Heinz Kunzelmann for his support, many helpful 
discussions and for making dental test images available.
This work benefited from the use of [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php), [bcolz](http://bcolz.blosc.org/en/latest), [numpy](https://numpy.org) [scipy](https://scipy.org/scipylib/index.html) and [cython](https://cython.org). The University of Applied Sciences, Augsburg, in particular the Faculty of 
Computer Science supported this project by granting sabbatical leaves.
Special thanks to Gisela Dachs, Andreas Gärtner, Evi Köbele,
Stefan König, Dominik Lüder, Thomas Obermeier and Sigrid Podratzky for acquiring test images and for keeping computers up and running.

# References
[1] T. Netsch,  P. Rösch,  A. v. Muiswinkel and J. Weese:
*Towards  Real-Time  Multi-Modality  3-D  Medical  Image  Registration.* Eight IEEE International Conference on Computer Vision, ICCV (2001) 718-725,</br>
[DOI: 10.1109/ICCV.2001.937595](https://ieeexplore.ieee.org/document/937595)
</br>
[2] P. Rösch and K.-H. Kunzelmann: *Efficient 3D rigid Registration of Large Micro CT Images.* International Journal of Computer assisted Radiology and Surgery **13 (Suppl. 1)** (2018) 118–119,</br> [DOI 10.1007/s11548-018-1766-y](https://doi.org/10.1007/s11548-018-1766-y)