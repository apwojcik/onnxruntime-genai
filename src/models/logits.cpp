// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "model.h"
#include "logits.h"

namespace Generators {

Logits::Logits(State& state)
    : state_{state},
      shape_{static_cast<int64_t>(state_.params_->BatchBeamSize()), model_.config_->model.vocab_size},
      type_{model_.session_info_->GetOutputDataType(model_.config_->model.decoder.outputs.logits)} {
  output_raw_ = std::make_unique<Tensor>(model_.p_device_inputs_, type_);

  if (model_.p_device_inputs_->GetType() == DeviceType::CUDA && !model_.config_->model.eos_token_ids.empty()) {
    auto& cpu_ids = model_.config_->model.eos_token_ids;
    cuda_eos_token_ids_ = model_.p_device_->Allocate<int32_t>(cpu_ids.size());
    copy(std::span<const int32_t>{cpu_ids}, cuda_eos_token_ids_.CpuSpan());
    cuda_eos_token_ids_.CopyCpuToDevice();
  }

  input_sequence_lengths.resize(state_.params_->search.batch_size);
}

DeviceSpan<float> Logits::Get() {
  size_t element_count = shape_[0] * shape_[1];

  // The model's output logits are {batch_size*num_beams, vocab_size}
  // OrtValue* logits_of_last_token = output_raw_->GetOrtTensor();
  OrtValue* logits_of_last_token = state_.outputs_[state_.output_names_.size() - 1];
  std::array<int64_t, 2> shape_last{shape_[0], shape_[1]};

  // Convert from float16 to float32 if necessary
  if (type_ == Ort::TypeToTensorType<Ort::Float16_t>) {
    Cast(*logits_of_last_token, logits_of_last_token_fp32_, *model_.p_device_inputs_, Ort::TypeToTensorType<float>);
    logits_of_last_token = logits_of_last_token_fp32_.get();
  }


  if (logits_.empty() || logits_of_last_token->GetTensorMutableRawData() != logits_.Span().data())
    logits_ = WrapTensor<float>(*model_.p_device_inputs_, *logits_of_last_token);


  // TODO: This functionality may have to be moved to DeviceInterface to make the code platform agnostic
  if (model_.p_device_inputs_->GetType() == DeviceType::CUDA) {
    if (!cuda_eos_token_ids_.empty())
      model_.p_device_inputs_->LaunchHandleEOSArray(
          logits_.Span().data(),
          static_cast<int>(shape_[0]) /* batch_beam_size*/,
          static_cast<int>(shape_[1]) /* vocab_size */,
          cuda_eos_token_ids_.Span().data(),
          static_cast<int>(cuda_eos_token_ids_.size()));
    return logits_;
  } else if (model_.p_device_inputs_->GetType() == DeviceType::DML) {
    HandleEOSArray(logits_.CopyDeviceToCpu());
    logits_.CopyCpuToDevice();
    return logits_;
  }

  HandleEOSArray(logits_.Span());
  return logits_;
}

void Logits::Update(const DeviceSpan<int32_t>& next_tokens, size_t new_kv_length) {
  if (output_raw_->ort_tensor_ && static_cast<size_t>(output_raw_->GetShape()[1]) == new_kv_length && new_kv_length == 1) {
    return;
  }

  // Store length of input sequence for each batch for the get step
  for (int b = 0; b < state_.params_->search.batch_size; b++) {
    // Find the first non pad token from the end
    size_t token_index = new_kv_length;
    while (token_index-- > 0) {
      auto next_token = const_cast<DeviceSpan<int32_t>&>(next_tokens).CpuSpan()[b * new_kv_length + token_index];
      if (next_token != model_.config_->model.pad_token_id)
        break;
    }
    input_sequence_lengths[b] = static_cast<int>(token_index + 1);
  }

  if (output_raw_->ort_tensor_ && static_cast<size_t>(output_raw_->GetShape()[1]) == new_kv_length) {
    return;
  }

  // shape_[1] = new_kv_length;
  output_raw_->CreateTensor(shape_, state_.params_->use_graph_capture);
  state_.outputs_[output_index_] = output_raw_->GetOrtTensor();
}

void Logits::HandleEOSArray(std::span<float> batched_logits) {
  if (model_.config_->model.eos_token_ids.empty())
    return;
 
    const size_t vocab_size = shape_[2];
  size_t vocab_index = 0;  // Simpler math to have this index go up by vocab_size for every logit chunk we process

  for (int index = 0; index < shape_[0]; index++) {
    auto logits = batched_logits.subspan(vocab_index, vocab_size);
    float max = std::numeric_limits<float>::lowest();
    for (auto id : model_.config_->model.eos_token_ids) {
      max = std::max(max, logits[id]);
      logits[id] = std::numeric_limits<float>::lowest();  // Set all EOS token options to never happen (the first will get the max of all)
    }

    logits[model_.config_->model.eos_token_id] = max;  // Set the score of the primary EOS token to the highest of any of the EOS tokens
    vocab_index += vocab_size;
  }
}

void Logits::Add() {
  output_index_ = state_.outputs_.size();

  state_.output_names_.push_back(model_.config_->model.decoder.outputs.logits.c_str());
  state_.outputs_.push_back(output_raw_->GetOrtTensor());
}

}  // namespace Generators
