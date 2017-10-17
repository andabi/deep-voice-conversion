#!/usr/bin/env bash

port=$1
case=$2

sh tensorboard_stop.sh $case
sh tensorboard_start.sh $port $case