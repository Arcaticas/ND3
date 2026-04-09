Setting up conda environment
The environment.yml should have all the dependencies:

conda env create -f environment.yml
conda activate td3-walker2d

May need to run this if breaking:
pip install torch numpy gymnasium[mujoco]

If all else fails consult the oracle.