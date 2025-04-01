## Continuous Batching

Example:

```cpp
#include "ort_genai.h"

std::queue<std::unique_ptr<OgaRequest>> request_queue;
int64_t request_id = GenerateUniqueRequestId();
request_queue.push(OgaRequest::Create(request_id, "What is 2 + 3?", OgaGeneratorParams::Create(...)))

auto config = OgaConfig::Create(config_path);
auto engine = OgaEngine::Create(*config);

while (!request_queue.empty()) {
    engine.AddRequest(std::move(request_queue.front()));
    request_queue.pop();
}

while (!engine.HasUnfinishedRequests()) {
    auto request_outputs = engine.Step();

    for (auto& request_output : request_outputs) {

        if (request_output.HasTokensToProcess()) {
            auto tokens = request_output.Tokens();
            auto text = request_output.Text();

            std::cout << "Request Id: " << request_output.RequestId() << std::endl;
            std::cout << "\tNew token: " << " " << tokens.back() << std::endl;
            std::cout << "\tGenerated text so far: " << text << std::endl;
        }

        if (request_output.Finished()) {
            engine.RemoveRequest(request_output.RequestId());
        }
    }
}
```

Overview:

- Once a new request is received from the client, it is added to the `request_queue`.
- The engine loop checks for unfinished requests in its internal pool. If there are unfinished requests, `Step` will generate the next round of tokens for requests it can serve.
- Requests that cannot be served (due to memory limitations) will continue sitting in the queue until the requests being currently served have finished. The key-value cache for finished requests can be moved to CPU while the unfinished ones can be moved into the device memory.
- At this point, the client can either remove the request from the engine or continue decoding the request by passing in a new request with the same Id.

## Components

- Engine
    - Scheduler
        - PagedKeyValueCache
            - CacheManager
- PagedAttention

## PagedAttention


## Block Based Key-Value Cache


## Limitations

- Continuous batching requests can be served for only a single model.
- LoRA adapters cannot be configured at the request level.

