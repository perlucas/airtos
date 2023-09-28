#! /bin/sh

# docker run -it --rm --user root -v "${PWD}":/home/jovyan/work \
#     -e GRANT_SUDO=yes \
#     -p 8888:8888 jupyter/tensorflow-notebook


docker compose start


# http://127.0.0.1:8888/lab?token=1833efe456be6216481d931b360d98f74b796a917c8ab388


# Descartados: MAB, SAC
# Posibles (discretizar el espacio de acciones): REINFORCE, PPO, DQN, C51

