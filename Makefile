# Create targets

.ONESHELL:
SHELL = /bin/bash

env :
	source /srv/conda/etc/profile.d/conda.sh
	conda env create -f environment.yml 
	conda activate notebook
	conda install ipykernel
	python -m ipykernel install --user --name make-env --display-name "IPython - Make"

html :
	jupyterbook build .
    
clean : 
	rm -rf figures/*
	rm -rf audio/*
	rm -rf _build/*
