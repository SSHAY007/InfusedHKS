# InfusedHKS

To build  
```
docker build -t hks -f ./scripts/Dockerfile .
```
The tag is "hks". The Dockerfile image is located in scripts. To run the dockerfile you have to expose your screen. Do not do this if you are in a shared user environment

```
docker run --mount type=bind,source="$PWD"/src,target=/root/code/baselines/src \
--rm \
-it \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=$DISPLAY \
hks2
```
