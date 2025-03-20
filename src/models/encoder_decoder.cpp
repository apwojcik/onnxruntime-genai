#include "../generators.h"
#include "encoder_decoder.h"

namespace Generators {
EncoderDecoderModel::EncoderDecoderModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  session_encoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.encoder.filename)).c_str(), session_options_.get());
  session_decoder_ = OrtSession::Create(ort_env, (config_->config_path / fs::path(config_->model.decoder.filename)).c_str(), session_options_.get());

  InitDeviceAllocator(*session_decoder_);
  session_info_->Add(*session_encoder_);
}

std::unique_ptr<State> EncoderDecoderModel::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<EncoderDecoderState>(*this, sequence_lengths, params);
}

EncoderState::EncoderState(const EncoderDecoderModel& model, RoamingArray<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      position_inputs_{model, *this, sequence_lengths_unk} {
  // Add audio features
  input_ids_.Add();
  position_inputs_.Add();                           // adds attention_mask

  // Add encoder hidden states
  auto hidden_states_shape = std::array<int64_t, 3>{static_cast<int>(input_ids_.GetShape()[0]), sequence_lengths, model_.config_->model.encoder.hidden_size};
  hidden_states_ = OrtValue::CreateTensor(*model_.allocator_device_, hidden_states_shape, float_type_);
  outputs_.push_back(hidden_states_.get());
  output_names_.push_back(model_.config_->model.encoder.outputs.hidden_states.c_str());
}

RoamingArray<float> EncoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
    int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
    State::Run(*model_.session_encoder_, *model_.run_options_, batch_size);
    return MakeDummy();
}

DecoderState::DecoderState(const EncoderDecoderModel& model, RoamingArray<int32_t> sequence_lengths_unk, const GeneratorParams& params)
    : State{params, model},
      model_{model},
      captured_graph_info_(model.GetCapturedGraphPool()->ReserveCapturedGraph(model, params)),
      position_inputs_{model, *this, sequence_lengths_unk} {
  input_ids_.Add();
  position_inputs_.Add();
  logits_.Add();
  extra_inputs_.Add();
  // hidden_states_.Add();

  // Add encoder hidden states
  auto hidden_states_shape = std::array<int64_t, 3>{static_cast<int>(input_ids_.GetShape()[0]), sequence_lengths, model_.config_->model.encoder.hidden_size};
  hidden_states_ = OrtValue::CreateTensor(*model_.allocator_device_, hidden_states_shape, float_type_);
  inputs_.push_back(hidden_states_.get());
  input_names_.push_back(model_.config_->model.decoder.inputs.hidden_states.c_str());

  // Add rnn states prev input
  auto rnn_states_prev_shape = std::array<int64_t, 3>{static_cast<int>3, static_cast<int>(input_ids_.GetShape()[0]), model_.config_->model.decoder.hidden_size};
  rnn_states_prev_ = OrtValue::CreateTensor(*model_.allocator_device_, rnn_states_prev_shape, float_type_);
  inputs_.push_back(rnn_states_prev_.get());
  input_names_.push_back(model_.config_->model.decoder.inputs.rnn_states_prev_.c_str());

  // Add rnn states output
  auto rnn_states_shape = std::array<int64_t, 3>{static_cast<int>3, static_cast<int>(input_ids_.GetShape()[0]), model_.config_->model.decoder.hidden_size};
  rnn_states_ = OrtValue::CreateTensor(*model_.allocator_device_, rnn_states_shape, float_type_);
  outputs.push_back(rnn_states_.get());
  output_names_.push_back(model_.config_->model.decoder.outputs.rnn_states_.c_str());
}

RoamingArray<float> DecoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  if (!first_run_) {
    UpdateInputsOutputs(next_tokens, next_indices, current_length);
  }

  int batch_size = static_cast<int>(input_ids_.GetShape()[0]);
  State::Run(*model_.session_decoder_, *model_.run_options_, batch_size);

  return logits_.Get();
}

void DecoderState::UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens_unk, RoamingArray<int32_t> beam_indices, int current_length) {
  input_ids_.Update(next_tokens_unk);
  position_inputs_.Update(current_length);
  rnn_states_prev_ = std::move(rnn_states_);
  logits_.Update();
}

EncoderDecoderState::EncoderDecoderState(const EncoderDecoderModel& model, const GeneratorParams& params, RoamingArray<int32_t> sequence_lengths_unk)
    : State{params, model},
      model_{model} {
  encoder_state_ = std::make_unique<EncoderState>(model, sequence_lengths_unk, params);
  decoder_state_ = std::make_unique<DecoderState>(model, sequence_lengths_unk, params);

}

RoamingArray<float> EncoderDecoderState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  if (encoder_state_->first_run_) {
    // Run encoder
    encoder_state_->Run(current_length, next_tokens, next_indices);

    // Run decoder
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);

    return logits;
  } else {
    first_run_ = false;
    // Update inputs and outputs for decoder
    decoder_state_->UpdateInputsOutputs(next_tokens, next_indices, current_length, first_run_);

    // Run decoder
    auto logits = decoder_state_->Run(current_length, next_tokens, next_indices);
    return logits;
  }
  // Not reached
  return MakeDummy();
}

OrtValue* EncoderDecoderState::GetInput(const char* name) {
  // Check if input name is in encoder state's inputs
  for (size_t i = 0; i < encoder_state_->input_names_.size(); i++) {
    if (std::strcmp(encoder_state_->input_names_[i], name) == 0) {
      return encoder_state_->inputs_[i];
    }
  }

  // Check if input name is in decoder state's inputs
  for (size_t i = 0; i < decoder_state_->input_names_.size(); i++) {
    if (std::strcmp(decoder_state_->input_names_[i], name) == 0) {
      return decoder_state_->inputs_[i];
    }
  }

  return State::GetInput(name);
};

OrtValue* EncoderDecoderState::GetOutput(const char* name) {
  // Check if output name is in encoder state's outputs
  for (size_t i = 0; i < encoder_state_->output_names_.size(); i++) {
    if (std::strcmp(encoder_state_->output_names_[i], name) == 0) {
      return encoder_state_->outputs_[i];
    }
  }

  // Check if output name is in decoder state's outputs
  for (size_t i = 0; i < decoder_state_->output_names_.size(); i++) {
    if (std::strcmp(decoder_state_->output_names_[i], name) == 0) {
      // Note: K caches will be transposed when returned
      return decoder_state_->outputs_[i];
    }
  }

  // cross_qk_final_ is an onnxruntime-genai maintained buffer that
  // is not part of the model's outputs, so we need to check for it here.
  if (std::strcmp("cross_qk", name) == 0) {
    return cross_qk_final_.get();
  }

  // cross_qk_search_buffer_ is an onnxruntime-genai maintained buffer that
  // is not part of the model's outputs, so we need to check for it here.
  if (std::strcmp("cross_qk_search", name) == 0) {
    return cross_qk_search_buffer_.get();
  }

  return State::GetOutput(name);
};

}  // namespace Generators
