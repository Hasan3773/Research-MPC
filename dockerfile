FROM carlasim/carla:0.9.13

USER root:root

RUN apt-get update -y || true && apt-get install python3-pip -y

USER carla:carla

CMD echo HelloWorld