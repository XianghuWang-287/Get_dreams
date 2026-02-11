build-docker:
	docker build -t mingdreams:latest .

run-docker:
	docker run  -it -v $(PWD)/data:/data mingdreams:latest /bin/bash