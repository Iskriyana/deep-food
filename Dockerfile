# get the ubuntu & cuda 10.1 & cudnn
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# get the python
RUN apt-get update && apt-get install -y \
	git \
	curl \
	ca-certificates \
	python3 \
	python3-pip \
	sudo \
	&& rm -rf /var/lib/apt/lists/*
	
RUN useradd -m gsidorov
RUN chown -R gsidorov:gsidorov /home/gsidorov/



#Get the GITHUB
RUN git clone https://github.com/Iskriyana/deep-food.git
RUN cd ./deep-food


#COPY THE BIG FILE
COPY  --chown=gsidorov /home/neuralcrypto/Desktop/DataScience_Projects/deep-food/models/final_model/variables/variables.data-00001-of-0000 /home/gsidorov/deep-food/models/final_model/variables



# CHECK MIHIS BUNDESTWEETS
# I THINK THAT you cna use the COPY. ./app I it is all that you eed
# There is a way of just uploading thisngs
# https://towardsdatascience.com/deploy-machine-learning-app-built-using-streamlit-and-pycaret-on-google-kubernetes-engine-fd7e393d99cb


# I think that I create the docker and then just export it!!
# https://stackoverflow.com/questions/28734086/how-to-move-docker-containers-between-different-hosts#:~:text=You%20cannot%20move%20a%20running,has%20created%20inside%20the%20container.



# install the reqs with pip 

RUN cd /home/gsidorov/deep-food/ && pip3 install -r requirements_deployment.txt




# The  last function that you will need to give to the docker or to the google cloud is something in the lines of what we
# had in the whatsception
# web: sh setup.sh && streamlit run app.py
