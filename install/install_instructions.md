## minIA installation

### Dockerfile 

Before you run the Dockerfile you need to download delf repository

```bash
# Model Garden for TensorFlow
git clone https://github.com/tensorflow/models.git
```
and then run:
```
docker build -t minia .
```

Now you can run our code using: 

```
docker run --runtime=nvidia -it minia:latest
```