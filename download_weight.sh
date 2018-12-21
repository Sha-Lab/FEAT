#!/usr/bin/env bash

function download() {
    printf "\033[32mstart to download $2 in $current/saves/$1\033[0m\n"
    [[ -e $1.zip ]] && rm -rf $1.zip
    wget -c https://doc-00-7o-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/jautjthvgoh3idbpvifflcpu1uo72846/1545379200000/09560182245773775633/*/1DFYbAta5mcMDtu1uW8f-8PFKsrHrTwJ0?e=download -O $1.zip
    if [[ -e $1 ]]; then
        backup=$1.`date '+%Y%m%d%H%M%S'`
        echo "backup current $1 directory first"
        rm -rf $1.*
        cp -rf $1 $backup
    fi
    unzip $1.zip
    rm -rf $1.zip
}

# We assume this will run under the FEAT folder
current=$(pwd)
[[ ! -e saves ]] && mkdir -p saves
cd saves

# Download pre-trained weights in "./saves/initialization".
download "initialization" "pre-trained weights"
echo ""
# Download learned models in "./saves/FEAT-Models"
download "FEAT-Models" "learned models"
