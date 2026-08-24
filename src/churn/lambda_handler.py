"""The AWS entry point: an ASGI adapter and nothing else.

Mangum translates an API Gateway HTTP API event into the ASGI scope FastAPI expects and
translates the response back. That is the whole module -- there is no churn logic here,
and no AWS anywhere else in the package, which is what lets the test suite run without a
single AWS import.

**`lifespan="off"` is deliberate.** ASGI lifespan is a server-lifecycle protocol: the
server runs startup once when it begins listening and shutdown when it stops. Lambda has
neither event. A container is created, frozen between invocations, and eventually killed
without warning, so a shutdown hook is a promise the platform does not keep. Leaving
lifespan on would also run startup inside the *first invocation* rather than the init
phase, putting any warm-up on a user's latency.

The warm-up therefore lives at module import in `api.py`: importing this module imports
`api`, which loads the artifact, and Lambda does that import during init -- before the
first request, on a clock that is billed separately from invocation. By the time this
handler is called the model is already resident.
"""

from mangum import Mangum

from churn.api import app

handler = Mangum(app, lifespan="off")
