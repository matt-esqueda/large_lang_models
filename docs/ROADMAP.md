# Project Roadmap & Working Notes

Living document. Update it in the same PR that changes anything here.

---

## 1. Status

**Active branch:** `fix/reproducible-setup`
**Phase:** hardening an existing prototype before restructuring.

| # | Branch | Scope | Status |
|---|--------|-------|--------|
| 1 | `fix/reproducible-setup` | clone reproducibility, requirements, line endings | in review |
| 2 | `fix/silent-correctness-bugs` | `_init_weights`, `self.device`, checkpoint interval, tokenizer | merged |
| 3 | `refactor/state-dict-checkpoints` | replace pickle with `state_dict` + optimizer state | in review |
| 4 | `refactor/shared-data-pipeline` | in-memory corpus, dedupe `resume_training.py` | planned |
| 5 | `test/core-invariants` | real pytest suite | planned |
| 6 | `refactor/package-layout` | installable package, CLI entry points, config wiring | planned |
| 7 | `docs/readme-accuracy` | README reconciliation | planned |

Correctness before restructuring. Migrating broken code produces
well-organized bugs.

---

## 2. Machines

Two machines, one GitHub account, one remote.

| Machine | GPU | torch build |
|---------|-----|-------------|
| Laptop (WSL2 / Ubuntu 22.04) | RTX 3060 Laptop, 6GB | `2.6.0+cu124` (stable) |
| Desktop | RTX 5080, 16GB | cu128 nightly (Blackwell requires it) |

Different torch builds on purpose - this is why `requirements.txt` does not
pin torch. Consequence: "works on my machine" carries less weight across
the pair. Verify anything version-sensitive on both.

### Before starting work on EITHER machine

```bash
git checkout main
git pull origin main
git checkout -b type/short-description
```

Always. After a squash merge the local branch history no longer matches
`main`, so branching off a stale local copy creates avoidable conflicts.

### Rules
- Never push the same branch from both machines. To move in-progress work:
  push the branch, pull it on the other machine, continue there only.
- Checkpoints and logs are gitignored and do NOT sync. Move manually or
  retrain.
- Clone into the Linux filesystem (`~/projects`), never `/mnt/c`. The 9p
  layer makes git and dataloader reads dramatically slower.
- Never install an NVIDIA driver inside WSL Ubuntu. The Windows driver is
  exposed through `/usr/lib/wsl/lib`; a Linux driver overwrites it and
  breaks CUDA. Verify passthrough with `nvidia-smi` inside WSL.

---

## 3. Environment setup (per machine, once)

conda is installed but used only to bootstrap venv (it supplies `ensurepip`,
which the `python3-venv` package would otherwise provide). The venv itself
is plain and conda-independent once created.

```bash
cd ~/projects/large_lang_models
conda activate base
python -m venv .venv
conda deactivate            # do NOT work inside conda base
source .venv/bin/activate
pip install --upgrade pip

# torch FIRST, matched to hardware:
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 5080 instead uses:
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

pip install -r requirements.txt
```

Verify the GPU actually executes kernels, not just that it is detected -
`torch.cuda.is_available()` can return True under WSL while launches fail:

```bash
python -c "
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(1000, 1000, device='cuda'); print((x @ x).shape)
"
```

### Disk space (WSL gotcha)

`df -h /` inside WSL reports the *virtual* disk, which can only grow by
claiming real space on Windows `C:`. A full C: breaks installs even though
WSL reports free space. Check both:

```bash
df -h /          # WSL view
df -h /mnt/c     # actual Windows drive
```

Budget ~6GB free on C: for a torch install (download + unpack + temp).
An interrupted torch install leaves every dependency installed but torch
itself missing - rebuild the venv rather than repairing it. The pip cache
also grows large (12GB observed); run `pip cache purge` after a successful
install.

---

## 4. Cross-machine invariant: the vocabulary

The character vocabulary derives from `data/raw/wizard_of_oz.txt`. If it
differs between machines, `lm_head` shapes mismatch and token IDs shift -
checkpoints silently become incompatible.

Current baseline: 80 characters (81 minus a stripped UTF-8 BOM).

Verify after any change to the corpus or to `prepare_data.py`:

```bash
sha256sum data/raw/wizard_of_oz.txt data/processed/vocab.txt
wc -l data/processed/vocab.txt
```

Hashes must match on both machines. `.gitattributes` forces LF so a CRLF
checkout cannot inject a carriage return into the character set.

---

## 5. Conventions

### Branches
`type/short-description` - `fix/`, `feat/`, `refactor/`, `test/`, `docs/`, `chore/`

### Commits
Conventional Commits:

```
type(scope): imperative summary under 72 chars

Body explaining WHY, not what. Wrap at 72.
```

### PRs
One logical change each. Squash merge, delete branch. `git log main` should
read as a changelog; a revert should undo a whole feature.

```bash
git push -u origin <branch>
gh pr create --fill
gh pr merge --squash --delete-branch
git checkout main && git pull origin main
```

### Definition of done
- [ ] Verified on a clean clone, not just the working copy
- [ ] Tests pass (once a suite exists)
- [ ] ROADMAP.md updated if status, decisions, or known issues changed
- [ ] README updated if user-facing behavior changed

---

## 6. Known issues (from audit)

Struck through = fixed.

### Blockers
- ~~`prepare_data.py` crashed on clean clone (`data/processed/` untracked)~~
- ~~`matplotlib` undeclared but imported by `plot_training.py`~~
- ~~`requirements.txt` pinned nightly wheels plus all transitive CUDA packages~~

### Correctness (PR 2)
- `_init_weights`: the `elif isinstance(module, nn.Embedding)` branch is
  nested inside the `nn.Linear` branch and is unreachable. Embeddings keep
  the default `N(0,1)` instead of `N(0,0.02)` - a 50x larger initial scale.
- `self.device` is a string captured at construction. `.to(device)` does not
  update it, so `torch.arange(T, device=self.device)` fails when a
  GPU-built model is loaded on CPU. Use `index.device`.
- `checkpoint_interval` is silently ignored unless it divides
  `eval_interval` - the save block is nested inside the eval block.

### Checkpoints (PR 3) - FIXED
- ~~`pickle.dump(model)` stores the class import path; renaming anything in
  `src/model.py` invalidates every existing checkpoint.
- ~~GPU-trained pickles cannot load on CPU (no `map_location` escape hatch).
- ~~`pickle.load` on an untrusted file executes arbitrary code.
- ~~`resume_training.py` builds a fresh `AdamW`, discarding moment estimates
  and causing a loss bump on resume.
- ~~Resume recovers the iteration number by regexing the filename; resuming
  from `model_final.pkl` silently restarts the counter at 0.

### Training quality (PR 4)
- Every batch re-reads the entire split file from disk, then slices ~2KB.
  One `estimate_loss()` call performs 200 full-file reads.
- All `batch_size` sequence starts are drawn from a single ~2100-char
  window, so "32 independent samples" are overlapping slices of two
  paragraphs.
- Post-norm blocks with no warmup, no LR schedule, no gradient clipping.
  Survivable at 6 layers, not at 12. Prefer pre-norm.
- No seeding anywhere - runs are not reproducible.
- `n_embd % n_head` unchecked; bad values fail deep in the residual add.
- Tokenizer `encode` raises `KeyError` on any out-of-vocab character. Most
  likely first-run failure for a new user typing a digit into `chat.py`.

### Structure (PR 5-7)
- `src/__init__.py` eagerly imports the model, so `prepare_data.py` cannot
  run without torch installed even though it only needs the tokenizer.
  Make the package import lazy.
- `resume_training.py` duplicates ~150 lines of `train.py` by copy-paste;
  the copies have already drifted on `block_size` handling.
- `test_training.py` is a smoke script, not a test suite. It asserts an
  exact checkpoint count, so it fails on any machine that has trained
  before. Non-idempotent.
- `chat.py` has two divergent generation paths; the streaming path silently
  drops `repetition_penalty` and reaches into `model._top_k_filtering`.
- `config/config.yaml` is dead - nothing imports yaml. Values are
  triplicated across the YAML, argparse defaults, and module constants.
- Seven scripts use `sys.path.append` instead of an installed package.
- Five scripts are undocumented in the README.
- Single-dash long flags (`-batch_size`) are non-standard; use `--batch-size`.
- `data/raw/wizard_of_oz.txt` is committed; should be a download script.

---

## 7. Decisions log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-09 | Do not pin torch in `requirements.txt` | One repo, two different hardware targets |
| 2026-09 | venv over conda env, bootstrapped from conda base | `python3-venv` absent; keeps pip workflow and matches README |
| 2026-09 | `makedirs` in code over `.gitkeep` | `.gitignore` has a bare `logs/` rule that would ignore a `.gitkeep` |
| 2026-09 | Correctness before restructuring | Restructuring touches every file; one clean diff set |
| 2026-09 | Squash merge | Readable, revertable `main` history |
| 2026-09 | `.pt` state_dict checkpoints over pickle | Survives refactors, loads cross-device, no arbitrary code execution |
| 2026-09 | Add ruff early (PR 4) | Repo has trailing whitespace on blank lines, which repeatedly broke exact-match patching |

---

## 8. Backlog (post-hardening)

- Branch protection on `main`: require passing CI, block direct pushes
- GitHub Actions: ruff + pytest on push and PR
- Pre-commit hooks
- LICENSE, CONTRIBUTING.md, CHANGELOG.md
- BPE tokenization to replace char-level
- Larger corpus with a download script
- Mixed precision and `torch.compile` on the 5080
- Reframe "chatbot": a base LM trained on next-token prediction over one
  book is a text continuer, not a chat model. Instruction tuning is a
  separate, much larger effort.
