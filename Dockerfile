FROM continuumio/miniconda3

# Set the ENTRYPOINT to use bash
# (this is also where you’d set SHELL,
# if your version of docker supports this)
ENTRYPOINT [ "/bin/bash", "-c" ]

# Conda supports delegating to pip to install dependencies
# that aren’t available in anaconda or need to be compiled
# for other reasons. In our case, we need psycopg compiled
# with SSL support. These commands install prereqs necessary
# to build psycopg.
RUN apt-get update && apt-get install -y \
 libpq-dev \
 build-essential \
&& rm -rf /var/lib/apt/lists/*

# Use the environment.yml to create the conda environment.
ADD asl_translator.yml /tmp/environment.yml
WORKDIR /tmp
RUN [ "conda", "env", "create" ]

ADD . /app

# Use bash to source our new environment for setting up
# private dependencies—note that /bin/bash is called in
# exec mode directly
WORKDIR /app

# We set ENTRYPOINT, so while we still use exec mode, we don’t
# explicitly call /bin/bash
CMD [ "source activate asl-od && exec python app.py" ]
