pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scipy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mujoco_py==2.1.2.14
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple stable-baselines3[extra]

# for gym error
pip install pip==24.0
pip install setuptools==65.5.0
pip install --user wheel==0.38.0
pip install gym==0.21.0

pip install -e .

# for mujoco_py error
pip uninstall cython
pip install cython==0.29.21
sudo apt-get install libosmesa6-dev
sudo apt-get install patchelf