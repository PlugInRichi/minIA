## minIA installation

### Dockerfile 
```bash
# From minIA/install/
docker build -t minia .
```

Now you can run our code using: 

```
docker run --runtime=nvidia --name imagenes_astronomicas -v ($pwd)/data:/data -v ($pwd)/src:/src --rm -it minia:latest
```

If you need using tensorborad or jupyter notebook you can start the container like this:

```
docker run --runtime=nvidia --name open_port_minIA -v $(pwd)/data:/data -v $(pwd)/src:/src -v $(pwd)/notebooks:/notebooks  -p 12000:11999 --rm -it minia:latest
```
