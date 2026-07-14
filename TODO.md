# TODO (open action items)

- [ ] **DEPTH-GEN (unblocks exp29 pose/flip benchmark + exp31 distance at scale):** MOVED TO
      LEONARDO. Home list already picked -> `scripts/depth_homes.txt` (20 cyclic held-out homes);
      `runs/hardneg/val_homes.txt` rebuilt from `results/heldout/heldout_ap.csv` (197 homes).
      Prereq: port DAP repo+weights to Leonardo at `$HOME/projects/DAP` (see trackD header), then
      `sbatch scripts/trackD_leonardo.slurm` (generates DAP depth for the 20 homes, then runs exp29).
      Claude: remind me until done.
- [ ] Train M2 distance head on data_floors (exp32) — Leonardo `sbatch scripts/trackC_leonardo.slurm`;
      runnable now (no depth needed). Compare val median dist vs exp31 DAP ~0.65 m.
- [ ] exp29 pose/flip benchmark once depth exists (bundled into trackD, or run standalone).
- [ ] **Argus (ECCV'26) baseline experiment:** run released Argus on ZInD one-pano-per-room; show it
      degrades as inter-room covisibility -> 0 while door-matching connectivity holds. Verify their
      overlap regime first (S C.4 "Scalability of Covisibility Module"). See ResearchLog 2026-07-14.
