// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "engine.h"

namespace Generators {



void Engine::Add(Request& request) {
  std::lock_guard<std::mutex> lock(added_requests_mutex_);
  requests_.push_back(request.shared_from_this());
}

void Engine::Remove(Request& request) {
  std::lock_guard<std::mutex> lock(removed_requests_mutex_);
  removed_requests_.push_back(request.shared_from_this());
}

Request* Engine::ProcessRequests() {
  std::lock_guard<std::mutex> lock(requests_mutex_);
  if (requests_.empty()) {
    return nullptr;
  }
  auto request = requests_.back().get();
  requests_.pop_back();
  return request;
}

}