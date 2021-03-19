if __name__ == "__main__":
    from setuptools import setup, find_packages, Extension
    from sys import platform

    try:
        from numpy.distutils.misc_util import get_numpy_include_dirs

        numpy_include_dirs = get_numpy_include_dirs()
    except ModuleNotFoundError:
        sys.exit("Please install numpy first")
    import lcreg

    if platform.startswith("win"):
        openmp_comp_args = [
            "/O2",
            "/openmp",
        ]
        openmp_link_args = [
            "/openmp",
        ]
    elif platform.startswith("linux"):
        openmp_comp_args = [
            "-DNDEBUG",
            "-O3",
            "-fopenmp",
        ]
        openmp_link_args = ["-fopenmp"]
    elif platform.startswith("darwin"):
        openmp_comp_args = [
            "-DNDEBUG",
            "-O3",
            "-fopenmp",
        ]
        openmp_link_args = ["-fopenmp", "-lgomp"]
    else:
        print("unsupported platform, please edit setup.py")
        exit(0)

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(
        name="lcreg",
        version=lcreg.__version__,
        description="Efficient 3D rigid and affine image registration",
        long_description_content_type="text/markdown",
        long_description=long_description,
        url="https://github.com/p-roesch/lcreg",
        author="Peter Rösch",
        author_email="lcreg@hs-augsburg.de",
        license="GPLv3",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "Programming Language :: Cython",
            "Programming Language :: Python :: 3 :: Only",
        ],
        keywords="3D image registration",
        project_urls={
            "ResearchGate Project": "https://www.researchgate.net/project/Efficient-registration-of-large-3D-images-lcreg",
        },
        packages=find_packages(exclude=["docs", "tests*"]),
        python_requires=">=3, <4",
        install_requires=[
            "numpy>=1.16",
            "scipy>=1.2",
            "bcolz>=1.2",
            "psutil>=5.6",
            "py-cpuinfo>=7.0",
        ],
        entry_points={
            "console_scripts": [
                "lcreg=lcreg.lcreg:main",
                "lcreg_profile=lcreg.lcreg:profiling_main",
                "view_compressed_images=lcreg.view_compressed_images:main",
                "compressed_to_mhd=lcreg.image3d:export_bcolz_image_main",
                "abs_difference_mhd=lcreg.image3d:abs_difference_mhd_main",
                "difference_mhd=lcreg.image3d:difference_mhd_main",
                "isq_to_mhd=lcreg.isq_to_mhd:main",
            ],
        },
        ext_modules=[
            Extension(
                "lcreg.lcreg_lib",
                ["lcreg/lcreg_lib.bycython.c"],
                extra_compile_args=openmp_comp_args,
                extra_link_args=openmp_link_args,
                include_dirs=numpy_include_dirs,
            ),
        ],
    )
