# FlashMLA

## code
```sh
├── csrc
│   ├── cutlass
│   ├── flash_api.cpp
│   ├── flash_fwd_mla_bf16_sm90.cu
│   ├── flash_fwd_mla_fp16_sm90.cu
│   ├── flash_fwd_mla_kernel.h
│   ├── flash_fwd_mla_metadata.cu
│   ├── flash_mla.h
│   ├── named_barrier.h
│   ├── softmax.h
│   ├── static_switch.h
│   └── utils.h
├── flash_mla
│   ├── __init__.py
│   └── flash_mla_interface.py
└── tests
    └── test_flash_mla.py
```

## start from python api
```mermaid
sequenceDiagram
    participant Python
    participant FlashAPTPython as flash_mla_inference.py
    participant FlashAPI as flash_api.cpp
    participant FlashFwdMLAMetaData as flash_fwd_mla_metadata.cu
    participant FlashFwdMLABF16SM90 as flash_fwd_mla_bf16_sm90.cu

    Python ->> FlashAPTPython: get_mla_metadata()
    FlashAPTPython ->> FlashAPI: get_mla_metadata()
    FlashAPI ->> FlashFwdMLAMetaData: get_mla_metadata_func()
    FlashFwdMLAMetaData ->> FlashFwdMLAMetaData: get_mla_metadata_kernel()
    FlashFwdMLAMetaData -->> FlashFwdMLAMetaData: return Mla_metadata_params
    FlashFwdMLAMetaData -->> FlashAPI: {tile_scheduler_metadata, num_splits}
    FlashAPI -->> Python: {tile_scheduler_metadata, num_splits}

    Python ->> FlashAPTPython: flash_mla_with_kvcache()
    FlashAPTPython ->> FlashAPI: mha_fwd_kvcache_mla()
    FlashAPI ->> FlashAPI: tensor checks and reshaping
    FlashAPI ->> FlashFwdMLABF16SM90: run_mha_fwd_splitkv_mla()
    FlashFwdMLABF16SM90 ->> FlashFwdMLABF16SM90: run_flash_splitkv_fwd_mla()
    FlashFwdMLABF16SM90 ->> FlashFwdMLABF16SM90: flash_fwd_splitkv_mla_kernel()
    FlashFwdMLABF16SM90 ->> FlashFwdMLABF16SM90: flash_fwd_splitkv_mla_combine_kernel()
    FlashFwdMLABF16SM90 -->> FlashAPI: {attention outputs, softmax logsumexp}
    FlashAPI -->> Python: {attention outputs, softmax logsumexp}
```