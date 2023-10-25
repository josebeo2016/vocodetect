# Supcon-voco

## Dir tree
- configs: yaml file for model variants
- datautils: dataloader classes
- docs: result txt files
- model: various model classes
- out: weighted parameters storage

## Configuration:
- *aug*_: anchor + aug real + other real + vocoded
- *augall*_: anchor + aug real + vocoded + aug vocoded
- *2_augall*_: anchor + aug real + other real + other aug real + vocoded + aug voco
- *3_augall*_: anchor + aug real + other real + vocoded + aug voco
- *4_augall*_: same to *3_augall* but support multigpu
- *5_augall*_: anchor + aug real + other real + vocoded + aug voco + other spoof

