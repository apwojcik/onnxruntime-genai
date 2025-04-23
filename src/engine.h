// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <mutex>

namespace Generators {

struct Request : std::enable_shared_from_this<Request>, LeakChecked<Request>, ExternalRefCounted<Request> {
  Request(TokenSequences& sequences, GeneratorParams& params) {
  }

  bool IsDone() const {
    return token_id_ == token_count_;
  }

  int32_t GetNextToken() {
    if (token_id_==token_count_)
      throw std::runtime_error("No more tokens to generate");
    return token_id_++;
  }

  void SetUserData(void* user_data) {
    user_data_ = user_data;
  }

  void* GetUserData() const {
    return user_data_;
  }

 private:
  void* user_data_{};
  int32_t token_id_ {};
  int32_t token_count_{10};
};

struct Engine : LeakChecked<Engine> {
  Engine(const Model& model) {}

  bool HasPendingRequests() const {
    return !requests_.empty() || !added_requests_.empty();
  }

  Request* ProcessRequests();

  void Add(Request& request);
  void Remove(Request& request);
  void Shutdown();

 private:
  std::mutex requests_mutex_;
  std::vector<std::shared_ptr<Request>> requests_;
  std::mutex added_requests_mutex_;
  std::vector<std::shared_ptr<Request>> added_requests_;
  std::mutex removed_requests_mutex_;
  std::vector<std::shared_ptr<Request>> removed_requests_;
};

}  // namespace Generators