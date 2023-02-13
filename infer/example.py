import ai

# using an MNIST model as an example
model = ai.examples.mnist.Model()

# spawn a worker process
inferencer = ai.infer.Inferencer(
    model,
    'cuda', # the worker will move the model to this device
    64, # the maximum inference batch size (will be less if there arent
        # sufficient requests available at the moment)
)

# the inferencer can be used as if it is the model
x = ai.randn(1, 1, 28, 28)
y1 = model(x)
y2 = inferencer(x)
assert (y1 == y2).all()

# update the parameters of the worker's model
inferencer.update_params(model.state_dict())

# you can also create an InferencerClient which can make inference requests but
# doesn't hold a reference to the worker (useful when passing it to other
# processes e.g. when data workers need to make inference requests)
client = inferencer.create_client()
y = client(x)

# requests can be made asynchronously
request_id = client.infer_async(x)
y = client.wait_for_resp(request_id)

# you can stop the worker directly via
del inferencer
# or you can just let `inferencer` go out of scope
