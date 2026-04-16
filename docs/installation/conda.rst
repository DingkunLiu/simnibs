.. _conda-install:

Install Using Conda (Advanced)
===============================

It is also possible to install SimNIBS using the `Conda <https://docs.conda.io/en/latest/>`_ package manager.

Windows
--------

1. Download and install the `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ **Python 3** distribution.

2. Locate the appropriate wheel for Windows on the release page (see `latest <https://github.com/simnibs/simnibs/releases/latest>`_), copy its address, and feed this directly to ``pip install`` as shown below

3. Download the `SimNIBS Windows environment file <https://github.com/simnibs/simnibs/releases/latest/download/environment_win.yml>`_

4. Open  *Anaconda Prompt*, which can be found in the *Start Menu*.

5. Run in the Prompt:

  .. code-block:: bash

      conda env create -f "%USERPROFILE%\Download\environment_win.yml"
      conda activate simnibs_env
      pip install https://github.com/simnibs/simnibs/releases/download/latest/simnibs-4.5.0-cp311-cp311-win_amd64.whl # change to the correct version

  \

6. (Optional) To setup the menu icons, file associations, the MATLAB library and add SimNIBS to the system path, run the :code:`postinstall_simnibs` script:

  .. code-block::

     md "%USERPROFILE%\SimNIBS"
     postinstall_simnibs --setup-links -d "%USERPROFILE%\SimNIBS"

  \

Linux
-------

1. Download and install the `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ **Python 3** distribution.

2. Locate the appropriate wheel for Linux on the release page (see `latest <https://github.com/simnibs/simnibs/releases/latest>`_), copy its address, and feed this directly to ``pip install`` as shown below

3. Download the `SimNIBS Linux environment file <https://github.com/simnibs/simnibs/releases/latest/download/environment_linux.yml>`_

4. Run in a terminal window:

  .. code-block:: bash

      export PATH="$HOME/miniconda/bin:$PATH" # This part can change depending on your miniconda installation
      conda env create -f ~/Downloads/environment_linux.yml
      conda activate simnibs_env
      pip install https://github.com/simnibs/simnibs/releases/download/v4.5.0/simnibs-4.5.0-cp311-cp311-linux_x86_64.whl # change to the correct version

  \

5. (Optional) To setup the menu icons, file associations, the MATLAB library and add SimNIBS to the system path, run the :code:`postinstall_simnibs` script:

  .. code-block:: bash

     mkdir $HOME/SimNIBS
     postinstall_simnibs --setup-links -d $HOME/SimNIBS

  \


MacOS
------

1. Download and install the `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ **Python 3** distribution.

2. Locate the appropriate wheel for MacOS on the release page (see `latest <https://github.com/simnibs/simnibs/releases/latest>`_), copy its address, and feed this directly to ``pip install`` as shown below

3. Download the `SimNIBS OSX environment file <https://github.com/simnibs/simnibs/releases/latest/download/environment_macOS.yml>`_

4. Run the following in a terminal window:

  .. code-block:: bash
  
      export PATH="$HOME/miniconda/bin:$PATH" # This part can change depending on your miniconda installation
      conda env create -f ~/Downloads/environment_macOS.yml
      conda activate simnibs_env
      pip install -f https://github.com/simnibs/simnibs/releases/download/v4.5.0/simnibs-4.5.0-cp311-cp311-macosx_11_0_arm64.whl
  
  \

5. (Optional) To setup the menu icons, file associations, the MATLAB library and add SimNIBS to the system path, run the :code:`postinstall_simnibs` script:

  .. code-block:: bash

     mkdir -p $HOME/Applications/SimNIBS
     postinstall_simnibs --setup-links -d $HOME/Applications/SimNIBS

  \
