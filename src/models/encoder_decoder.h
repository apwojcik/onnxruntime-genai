#pragma once
#include "model.h"
#include "input_ids.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"
#include "extra_inputs.h"

namespace Generators {

struct EncoderDecoderModel : Model {
  EncoderDecoderModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_decoder_;
  std::unique_ptr<OrtSession> session_encoder_;
};

struct EncoderState : State {
    EncoderState(const EncoderDecoderModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
    RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;
    const CapturedGraphInfo* GetCapturedGraphInfo() const override { return captured_graph_info_.get(); };
  
   private:
    void UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length);
  
    const EncoderDecoderModel& model_;
    CapturedGraphInfoPtr captured_graph_info_;
  
    InputIDs input_ids_{*this};
    PositionInputs position_inputs_;
    ExtraInputs extra_inputs_{*this};
  };

struct DecoderState : State {
  DecoderState(const EncoderDecoderModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;
  const CapturedGraphInfo* GetCapturedGraphInfo() const override { return captured_graph_info_.get(); };

 private:
  void UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length);

  const EncoderDecoderModel& model_;
  CapturedGraphInfoPtr captured_graph_info_;

  InputIDs input_ids_{*this};
  Logits logits_{*this};
  PositionInputs position_inputs_;
  ExtraInputs extra_inputs_{*this};
};

struct EncoderDecoderState : State {
    EncoderDecoderState(const EncoderDecoderModel& model, const GeneratorParams& params, RoamingArray<int32_t> sequence_lengths);
    EncoderDecoderState(const EncoderDecoderState&) = delete;
    EncoderDecoderState& operator=(const EncoderDecoderState&) = delete;
  
    RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;
    OrtValue* GetInput(const char* name) override;
    OrtValue* GetOutput(const char* name) override;
  
  private:
    void UpdateInputsOutputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length);
  
    const EncoderDecoderModel& model_;
  
    std::unique_ptr<EncoderState> encoder_state_;
    std::unique_ptr<DecoderState> decoder_state_;
  };

}  // namespace Generators
