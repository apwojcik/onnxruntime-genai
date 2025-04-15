// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model_executor.h"
#include "decoders/simple_decoder.h"

#include <typeinfo>

namespace Generators {

namespace {

std::unique_ptr<Decoder> CreateDecoder(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager) {
  if (auto decoder_only_model = std::dynamic_pointer_cast<DecoderOnly_Model>(model)) {
    return std::make_unique<SimpleDecoder>(decoder_only_model, cache_manager);
  } else {
    throw std::runtime_error("Continuous batching does not support the requested model type.");
  }
  return nullptr;
}

}  // namespace

ModelExecutor::ModelExecutor(std::shared_ptr<Model> model, std::shared_ptr<CacheManager> cache_manager)
    : model_{model},
      decoder_{CreateDecoder(model, cache_manager)} {}

void ModelExecutor::Decode(ScheduledRequests& scheduled_requests) {
  decoder_->Decode(scheduled_requests);
}

}  // namespace Generators
