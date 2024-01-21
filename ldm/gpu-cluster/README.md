Build the docker image with

```
docker build -t dissertation-cluster-image --build-arg requirements="$(cat ../requirements.txt | base64 -w 0)" .
```