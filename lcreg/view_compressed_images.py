# lcreg: Efficent rigid and affine 3D image registration
#
# Copyright (C) 2019  Peter Rösch, Peter.Roesch@hs-augsburg.de
#
# Organisation:
# Faculty of Computer Science, Augsburg University of Applied Sciences,
# An der Hochschule 1, 86161 Augsburg, Germany
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python

"""
convert compressed image to mhd and view the result with itkSNAP
"""
import sys
import os
from subprocess import call
import tempfile
import shutil
from lcreg import image3d


def main():
    #
    # usage message
    if len(sys.argv) < 3:
        print("usage: view_compressed_images tmp_prefix path_name_1 ", end="")
        print(" path_name_2 ...")
        return
    #
    # check for valid temporary path
    root_tmp_path = sys.argv[1]
    if root_tmp_path[-1] != os.path.sep:
        root_tmp_path += os.path.sep
    tmp_path = tempfile.mkdtemp(prefix=root_tmp_path)
    #
    # export bcolz images to mhd format and remember mhd file names
    mhd_file_names = []
    for bcolz_name in sys.argv[2:]:
        while bcolz_name[-1] == os.path.sep:
            bcolz_name = bcolz_name[:-1]
        mhd_file_names.append(
            os.path.join(tmp_path, os.path.basename(bcolz_name) + ".mhd")
        )
        try:
            image3d.export_bcolz_image(bcolz_name, mhd_file_names[-1], True)
        except Exception as e:
            print(str(e))
            return
    #
    # set up command and call it
    if sys.platform == "darwin":
        MAC_OS_SNAP_ENVIRON = "ITKSNAP_APP_DIR"
        if MAC_OS_SNAP_ENVIRON in os.environ:
            snap_exe = (
                os.environ[MAC_OS_SNAP_ENVIRON]
                + os.sep
                + "ITK-SNAP.app/Contents/MacOS/ITK-SNAP"
            )
        else:
            print(
                f"\nRequired environment variable {MAC_OS_SNAP_ENVIRON}"
                " not set."
            )
            print("Exiting ...\n")
            sys.exit(-1)
    elif sys.platform.startswith("win"):
        snap_exe = "ITK-SNAP"
    else:
        snap_exe = "itksnap"

    cmd = snap_exe + " -g " + mhd_file_names[0]
    if len(mhd_file_names) > 1:
        cmd += " -o"
    for fn in mhd_file_names[1:]:
        cmd += " " + fn
    call(cmd.split())
    #
    # cleanup
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()
