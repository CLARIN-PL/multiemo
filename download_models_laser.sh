#!/bin/bash

if ! test -d "./models/laser_bilstm"; then
	mkdir -p ./models/laser_bilstm
	wget https://minio.clarin-pl.eu/public/models/multiemodetect/models_laser_bilstm.7z -O models/laser_bilstm/models_laser_bilstm.7z
	7z x -omodels/laser_bilstm models/laser_bilstm/models_laser_bilstm.7z
	rm models/laser_bilstm/models_laser_bilstm.7z
fi

