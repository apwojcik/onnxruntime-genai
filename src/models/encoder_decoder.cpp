// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "encoder_decoder.h"
#include <vector>
#include "../sequences.h"

namespace Generators {

EncoderDecoderModel::EncoderDecoderModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.encoder_decoder_init.filename)).c_str(), session_options_.get());
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
  session_info_->Add(*session_encoder_);
}

std::unique_ptr<State> EncoderDecoderModel::CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<EncoderDecoderState>(*this, sequence_lengths, params);
}

EncoderDecoderState::EncoderDecoderState(const EncoderDecoderModel& model, DeviceSpan<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      encoder_attention_mask_{model, *this, sequence_lengths_unk}  {
}

DeviceSpan<float> EncoderDecoderState::Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) {
  if(first_run_) {
    //INITIALIZE THE ENCODER AND RUN IT ONCE

    // std::cout<<"Initializing encoder"<<std::endl;
    encoder_input_ids_.name_ = "encoder_input_ids";
    encoder_input_ids_.Add();
    // auto& stream = Log("Encoder Input IDS");
    // stream << std::endl;
    // DumpTensors(model_, stream, inputs_.data(), input_names_.data(), input_names_.size(), true);
    // std::cout<<"Added encoder input ids"<<std::endl;

    encoder_attention_mask_.Add();
    // std::cout<<"Next Token size = "<<next_tokens.size()<<std::endl;
    
    cross_cache_ = std::make_unique<CrossCache>(*this, next_tokens.size());
    AddEncoderCrossCache(cross_cache_);
  
    encoder_input_ids_.Update(next_tokens);
    size_t new_length = static_cast<size_t>(encoder_input_ids_.GetShape()[1]);
    encoder_attention_mask_.Update(next_tokens, current_length, static_cast<int>(new_length));
    State::Run(*model_.session_encoder_);
    std::cout<<"Encoder has been run"<<std::endl;

    // CLEAR INPUTS AND OUTPUTS
    ClearIO();

    //INITIALIZE THE DECODER
    std::cout<<"Initializing decoder"<<std::endl;
    // const std::array<int64_t, 2> zero_sized_tensor{params_->BatchBeamSize(), 1};
    // std::cout<<"Zero sized tensor = "<<zero_sized_tensor[0]<<" "<<zero_sized_tensor[1]<<std::endl;
    // zero_tensor = OrtValue::CreateTensor(model_.allocator_cpu_, zero_sized_tensor, model_.session_info_->GetInputDataType(model_.config_->model.decoder.inputs.input_ids));
    // for(int i=0;i<params_->BatchBeamSize();i++) {
    //   zero_tensor->GetTensorMutableData<int32_t>()[i] = 0;
    // }
    // *zero_tensor->GetTensorMutableData<int32_t>() = 0;
    input_ids_.name_ = "input_ids";
    // input_ids_.Add();
    input_ids_.AddDecoderInputs();
    // input_ids_.Update(empty_span);
    // std::cout<<"Added input ids"<<std::endl;
    // input_names_.push_back("input_ids");
    // inputs_.push_back(zero_tensor.get());
    // std::cout<<"Added input names"<<std::endl;

    encoder_attention_mask_.Add();

    logits_.Add();
    // std::cout<<"Added logits"<<std::endl;
    kv_cache_.Add();
    // new_length = zero_sized_tensor[1];
    new_length = static_cast<size_t>(input_ids_.GetShape()[1]);
    kv_cache_.Update(next_indices, new_length);
    logits_.Update(next_tokens, new_length);
    // std::cout<<"Added kv cache"<<std::endl;

    AddDecoderCrossCache(cross_cache_);
    // std::cout<<"FIRST RUN"<<std::endl;

    State::Run(*model_.session_decoder_);
    std::cout<<"RUNNING DECODER"<<std::endl;
    first_run_ = false;
    std::cout<<"In the first run next token size is = "<<next_tokens.size()<<std::endl;
    return logits_.Get();
  }

    // UPDATE THE DECODER
    std::cout<<"Updating input ids with next tokens = "<<next_tokens.size()<<std::endl;
    input_ids_.Update(next_tokens);
    // input_ids_.UpdateDecoder(next_tokens);

    size_t new_length = static_cast<size_t>(input_ids_.GetShape()[1]);
    encoder_attention_mask_.Update(next_tokens, current_length, static_cast<int>(new_length));
    std::cout<<"Updated encoder attention mask = "<<current_length<<std::endl;
    std::cout<<"New length = "<<new_length<<std::endl;

    kv_cache_.Update(next_indices, new_length + 1);

    logits_.Update(next_tokens, new_length);
    // std::cout<<"Updated logits"<<std::endl;

    // RUN THE DECODER
    State::Run(*model_.session_decoder_);
    std::cout<<"RUNNING DECODER"<<std::endl;
    return logits_.Get();

}

}  // namespace Generators
