# To generate DreaMS embeddings:


``` 
# if DreaMS doesn't exist in this directory yet
git clone https://github.com/pluskal-lab/DreaMS.git 
cd DreaMS

# Create conda environment
conda create -n dreams_env python==3.11.0 --yes
conda activate dreams_env

# Install DreaMS
pip install -e .
```

Then 
```
cd ..
pip install chardet
python3 dreams_from_mgf.py input_mgf_path.mgf embeddings_output.parquet
```

Then DreaMS will download its embedding model if it isnt downloaded yet.


## Run it in docker

```
make build-docker
```

Then you can run the container with:

```
make run-docker
``` 

```
python3 dreams_from_mgf.py input_mgf_path.mgf embeddings_output.parquet
```