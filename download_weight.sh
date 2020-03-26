#!/usr/bin/env bash

function download() {
    printf "\033[32mstart to download $2 in $current/saves/$1\033[0m\n"
    [[ -e $1.zip ]] && rm -rf $1.zip
    wget -c https://drive.google.com/uc?export=download&confirm=wWiE&id=1XcUZMNTQ-79_2AkNG3E04zh6bDYnPAMY -O $1.zip
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