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

    std::vector<std::list<std::unique_ptr<OgaRequest>>::iterator> requests_to_remove;

    for (auto request_it = request_pool.begin(); request_it != request_pool.end(); ++request_it) {
        if (auto tokens = request.GetNewTokens()) {
            for (auto token : tokens) {
                std::cout << "New token: " << " " << token << std::endl;
            }

            if (request->Done()) {
                request->Remove();
                requests_to_remove.push_back(request_it);
            }
        }    
    }

    for (auto& it : requests_to_remove) {
        request_pool.erase(it);
    }
}
```

Overview:

- Once a new request is received from the client, it is added to the `request_queue`.
- The engine loop checks for unfinished requests in its internal pool. If there are unfinished requests, `Step` will generate the next round of tokens for requests it can serve.
- Requests that cannot be served (due to memory limitations) will continue sitting in the queue until the requests being currently served have finished. The key-value cache for finished requests can be moved to CPU while the unfinished ones can be moved into the device memory.
- At this point, the client can either remove the request from the engine or continue decoding the request by adding tokens to an existing request.

## Components

- Rewire Onnx Runtime GenAI infrastructure to handle new components:
    - Engine, Scheduler, PagedKeyValueCache, CacheManager, Request, SequenceGroup
- PagedAttention custom operator (Cuda)
- ONNX Runtime GenAI C/C++/C#/Python API
- Prepare model inputs (serialized)
- Process model outputs (logits)
- Search/Sampling
- Tokenize/Detokenize input/output internally and bundle the output into the request
- Prepare model with updated op and input/output structure


## PagedAttention


## Block Based Key-Value Cache


## Limitations

- Continuous batching requests can be served for only a single model.
- LoRA adapters cannot be configured at the request level.

