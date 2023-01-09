#conda env remove -n pointmae_environment

conda env create -f scripts/environment.yml

# not necessary just for classification
#cd ./extensions/chamfer_dist
#python setup.py install --user
#cd ../..
# optional for data folder sharing
# ln -s ../data data


