// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(_WIN32)

#include <Windows.h>
#include <dbghelp.h>
#include <memory>
#include <string>
#include <filesystem>
#include "string.h"
#pragma comment(lib, "dbghelp.lib")

struct HMODULE_Deleter {
  typedef HMODULE pointer;
  void operator()(HMODULE h) { FreeLibrary(h); }
};

using ModulePtr = std::unique_ptr<HMODULE, HMODULE_Deleter>;

String DetermineLoadLibraryError(const std::filesystem::path& filename) {
  String error(__TEXT("Error loading"));

  ModulePtr hModule;  // Here so that filename is valid until the next iteration
    error += __TEXT(" \"") + filename.to_native() + __TEXT("\"");

  // We use DONT_RESOLVE_DLL_REFERENCES instead of LOAD_LIBRARY_AS_DATAFILE because the latter will not process the import table
  // and will result in the IMAGE_IMPORT_DESCRIPTOR table names being uninitialized.
  hModule = ModulePtr{LoadLibraryEx(filename.to_native().c_str(), nullptr, DONT_RESOLVE_DLL_REFERENCES)};
  if (!hModule) {
    error += __TEXT(" which is missing. (Error ") + ToString(GetLastError()) + __TEXT(')');
    return error;
  }

  // Get the address of the Import Directory
  ULONG size{};
  const auto* import_desc = static_cast<IMAGE_IMPORT_DESCRIPTOR*>(ImageDirectoryEntryToData(hModule.get(), FALSE, IMAGE_DIRECTORY_ENTRY_IMPORT, &size));
  if (!import_desc) {
    error += __TEXT(" No import directory found.");  // This is unexpected, and I'm not sure how it could happen, but we handle it just in case.
    return error;
  }

  // Iterate through the import descriptors to see which dependent DLL can't load
  for (; import_desc->Characteristics; import_desc++) {
    auto* dll_name = reinterpret_cast<TCHAR*>(reinterpret_cast<BYTE*>(hModule.get()) + import_desc->Name);
    // Try to load the dependent DLL, and if it fails, we loop again with this as the DLL and we'll be one step closer to the missing file.
    ModulePtr hDepModule{LoadLibrary(dll_name)};
    if (!hDepModule) {
      error += __TEXT(" which depends on");
      break;
    }
  }

  error += __TEXT(" But no dependency issue could be determined.");
  return error;
}

#endif
