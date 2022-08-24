## minIA installation

### Dockerfile 
```bash
# From minIA/install/
docker build -t minia .
```

Now you can run our code using: 

```
docker run --runtime=nvidia -it minia:latest
```
