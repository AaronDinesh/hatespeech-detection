#####################################
# Base image with GPU + PyTorch support
#####################################
FROM nvcr.io/nvidia/ai-workbench/pytorch:1.0.6

#####################################
# RCP CaaS: User mapping for storage access
#####################################
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

RUN groupadd -g ${LDAP_GID} ${LDAP_GROUPNAME} && \
    useradd -m -s /bin/bash -u ${LDAP_UID} -g ${LDAP_GID} ${LDAP_USERNAME}

#####################################
# Set up workspace and copy files
#####################################
WORKDIR /home/${LDAP_USERNAME}

COPY --chown=${LDAP_UID}:${LDAP_GID} ./ /home/${LDAP_USERNAME}/llm_detection


#####################################
# Install system dependencies (minimized)
#####################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

#####################################
# Install Python dependencies
#####################################
USER ${LDAP_USERNAME}
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /home/${LDAP_USERNAME}/llm_detection/requirements.txt

#####################################
# Ensure local bin is on PATH (safe fallback)
#####################################
ENV PATH=$PATH:/home/${LDAP_USERNAME}/.local/bin
