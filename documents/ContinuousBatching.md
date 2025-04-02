## Continuous Batching

Example:

```cpp
#include "ort_genai.h"

std::queue<std::string> request_queue;
request_queue.push("What is 2 + 3?");

std::list<std::unique_ptr<OgaRequest>> request_pool;

auto config = OgaConfig::Create(config_path);
auto engine = OgaEngine::Create(*config);

while (!request_queue.empty()) {
    request_pool.push_back(OgaRequest::Create(
        request_queue.front().prompt(), OgaGeneratorParams::Create(...)));
    engine.AddRequest(request_pool.back());
    request_queue.pop();
}

while (engine.HasPendingRequests()) {
    engine.Step();

    for (auto request_it = request_pool.begin(); request_it != request_pool.end();) {
        if (request->HasAvailableTokens()) {
            auto tokens = request->Tokens();
            auto text = request->Text();

            std::cout << "New token: " << " " << tokens.back() << std::endl;
            std::cout << "Generated text so far: " << text << std::endl;
        }

        if (request->Done()) {
            request->Remove();
            request_pool.erase(request_it++);
        } else {
            request_it++;
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

