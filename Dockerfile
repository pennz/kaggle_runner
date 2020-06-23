FROM debian:buster
LABEL maintainer="fireflysuccess@gmail.com"

ENV TOOLS 'rsync curl tig ctags htop tree pv nmap screen time tmux netcat psmisc vim neovim ca-certificates fish' 

RUN apt-get update && apt-get install -y ${TOOLS} --no-install-recommends
    
RUN mkdir /content
COPY .* /content/
WORKDIR /content
COPY .git/ ./.git/

RUN make
ENTRYPOINT ["/usr/bin/vim"]
