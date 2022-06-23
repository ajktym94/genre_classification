yes | conda create --name myenv
conda activate myenv
yes | conda install pytorch torchvision torchaudio torchtext cudatoolkit=11.3 -c pytorch
yes | conda install ipykernel pandas seaborn
ipython kernel install --user --name=myenv
yes | conda install -c anaconda scikit-learn gensim
yes | conda install -c conda-forge matplotlib ipywidgets