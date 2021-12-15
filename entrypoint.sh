#!/bin/bash

if ! test -d "./models"; then
	mkdir -p ./models
	wget https://minio.clarin-pl.eu/public/models/multiemodetect/models.7z -O models/models.7z
	7z x -omodels models/models.7z
	rm models/models.7z
fi

uvicorn api:app --host 0.0.0.0 --port 8744 --workers $API_WORKERS
