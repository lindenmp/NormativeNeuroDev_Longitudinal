# neurodev_long
Longitudinal normative modelling project on PNC data

# Environment build

    conda create -n neurodev_long python=3.7
    conda activate neurodev_long

    # Essentials
    pip install jupyterlab ipython pandas numpy seaborn matplotlib nibabel glob3
    pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install
    
	# Statistics
	pip install scipy statsmodels sklearn pingouin

	# Pysurfer for plotting
	pip install mayavi
	pip install PyQt5
	jupyter nbextension install --py mayavi --user
	jupyter nbextension enable --py mayavi --user
	pip install pysurfer

    cd /Users/lindenmp/Dropbox/Work/ResProjects/neurodev_long
    conda env export > environment.yml
	pip freeze > requirements.txt

