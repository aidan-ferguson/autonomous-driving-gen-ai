Build the docker image with

```
docker build -t aidanferguson1/dissertation:latest --build-arg requirements="$(cat ../requirements.txt | base64 -w 0)" .
```