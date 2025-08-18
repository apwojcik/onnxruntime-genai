// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>
#include <codecvt>

#ifndef TCHAR
#if defined(_UNICODE) || defined(UNICODE)
#define TCHAR   wchar_t
#else
#define TCHAR   char
#endif
#endif

#if defined(UNICODE) || defined(_UNICODE)
using String = std::wstring;
#define ToString std::to_wstring
#define to_native native
inline std::string WideToUTF8String(std::wstring_view s) {
  std::wstring_convert<std::codecvt_utf8<TCHAR>> converter;
  return converter.to_bytes(s.data());
}
#else
using String = std::string;
#define ToString std::to_string
#define to_native string
inline std::string WideToUTF8String(std::string_view s) {
  return std::string{s};
}
#endif
