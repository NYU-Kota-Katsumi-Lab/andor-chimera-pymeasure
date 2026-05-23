# CLAUDE.md

Project-specific guidance for Claude Code when working on andor-chimera-pymeasure.

## CRITICAL: Background Process Management

**NEVER run more than 1-2 background Bash commands at a time.** This machine has limited memory. Excessive background processes (especially Python/pytest) cause Claude Code itself to crash.

Rules:
- **Run tests in FOREGROUND** — never use `run_in_background` for pytest or `uv run` commands
- **Only use background for**: launching the app (`uv run python -m andor_qt`) or long git operations
- **Before launching a new background process**, check if previous ones are still running
- **Agent subprocesses are fine** — they manage their own lifecycle
- If a foreground command takes >2 minutes, increase the timeout instead of backgrounding it

**Killing Python processes (MUST do before relaunching the app):**
```bash
# Use PowerShell — most reliable on this machine
powershell.exe -Command "Stop-Process -Name python -Force -ErrorAction SilentlyContinue"
# Wait for camera SDK to release
sleep 3
```
**ALWAYS kill before relaunch.** The Andor camera SDK locks exclusively — if a previous python.exe holds it, the new instance gets DRV_NOT_AVAILABLE (20992).

## CRITICAL: Development Workflow

**STOP AND PLAN BEFORE CODING.** This project requires strict TDD with atomic commits.

### Required Workflow for Every Feature

1. **Plan atomic commits FIRST** - Before writing any code, identify the commits:
   ```
   Task: "Add user feedback to sequencer"

   Commits:
   1. feat: add FeedbackSequencerWidget class
   2. feat: integrate FeedbackSequencerWidget into main window
   3. test: add validation feedback tests
   ```

2. **For each commit, follow TDD**:
   - **RED**: Write failing tests first
   - Run tests → verify they FAIL
   - **GREEN**: Write minimal code to pass
   - Run tests → verify they PASS
   - **COMMIT**: Create atomic commit immediately

3. **Use task tracking** to maintain discipline:
   ```
   TaskCreate → for each planned commit
   TaskUpdate(in_progress) → when starting
   TaskUpdate(completed) → after commit
   ```

### What NOT To Do

❌ Implement features first, then write tests
❌ Make multiple changes before committing
❌ Batch all commits at the end
❌ Skip the planning phase

### Commit Checklist

Before each commit, verify:
- [ ] Tests were written BEFORE implementation
- [ ] Tests failed before implementation (RED phase verified)
- [ ] Commit does exactly ONE thing
- [ ] Tests pass after this commit
- [ ] No unrelated changes included

### Agents to Use

| Phase | Agent | Purpose |
|-------|-------|---------|
| Planning | `planner` or `/plan` | Break feature into atomic commits |
| Architecture | `architect` | Review design before implementation |
| Testing | `tdd-guide` | Write tests first (RED phase) |
| Implementation | Direct coding | Write minimal code to pass tests |
| Commit | `git-commit` | Verify atomic, create commit |
| Debug | `debugger` | If tests fail unexpectedly |

### Example Workflow

```
User: "Add wavelength validation to spectrograph"

1. PLAN (use planner agent):
   Commits:
   - feat: add wavelength range validation to MockSpectrograph
   - feat: add validation error feedback in SpectrographControlWidget

2. FOR EACH COMMIT:
   a. TaskCreate("Add wavelength validation to MockSpectrograph")
   b. TaskUpdate(status=in_progress)
   c. Write test: test_wavelength_rejects_out_of_range
   d. Run pytest → FAILS (RED)
   e. Implement validation in MockSpectrograph
   f. Run pytest → PASSES (GREEN)
   g. git commit
   h. TaskUpdate(status=completed)
   i. Move to next commit
```

## Project Overview

Qt/PySide6 GUI and PyMeasure integration for Andor CCD spectrometer control. Two packages:
- **andor_qt** — Main GUI application
- **andor_pymeasure** — PyMeasure procedures and instrument drivers

## Project Stack

- **Language**: Python 3.11+
- **Build system**: uv with pyproject.toml
- **GUI framework**: PySide6 (Qt6)
- **Experiment framework**: PyMeasure
- **Testing**: pytest with pytest-qt
- **Linting**: ruff

## Code Style Guidelines

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with these project-specific overrides:

- **Line length**: 100 characters (not 80 — Qt code is verbose)
- **Naming**: `lower_with_under` for functions/variables, `CamelCase` for classes, `CAPS_WITH_UNDER` for constants
- **Imports**: Absolute imports only, sorted by ruff, one per line (except `typing`)
- **Type hints**: Required on all public API functions. Use `from __future__ import annotations`
- **Docstrings**: Google style with `Args:`, `Returns:`, `Raises:` sections. Start with imperative verb
- **Mutable defaults**: Never use mutable objects (`[]`, `{}`) as default arguments
- **None checks**: Always `is None` / `is not None`, never `== None`
- **Boolean eval**: Use implicit false for sequences (`if not items:`)
- **Exceptions**: Never bare `except:`; minimize code inside `try` blocks
- **Properties**: Use for controlled access; no surprising side effects

### Property Docstrings (PyMeasure style)

For instrument properties, follow PyMeasure conventions:
```python
# Use "Control", "Measure", "Get", or "Set" to indicate property type
wavelength = Instrument.control(
    "?GW", ":GW %g",
    """Control the center wavelength in nm (float from 200 to 1100).""",
    validator=strict_range,
    values=[200, 1100],
)

temperature = Instrument.measurement(
    "?TEMP",
    """Measure the current temperature in degrees C (float, read-only).""",
)
```

## Quick Commands

```bash
# Run in mock mode (no hardware)
uv run python -m andor_qt --mock

# Run full test suite (exclude hardware integration & e2e)
uv run pytest tests/ --ignore=tests/integration --ignore=tests/e2e -q --tb=short

# Run specific test file
uv run pytest tests/qt/test_spectrum_overlay.py -v

# Run tests without uv (avoids uv package-sync overhead)
.venv/Scripts/pytest.exe tests/ --ignore=tests/integration --ignore=tests/e2e -q --tb=short

# Check formatting
uv run ruff check src/
```

### Test Execution Notes

- **Always exclude `tests/integration` and `tests/e2e`** — these require real hardware
- **Prefer `.venv/Scripts/pytest.exe` directly** when iterating fast — `uv run` syncs packages each time
- **Run Qt-heavy test files individually** if you see hangs — combining many Qt test files in one process can deadlock the event loop
- **Background mode causes hangs** — always run pytest in foreground

## Andor SDK Notes

### SDK Location
- **Windows**: `C:\Program Files\Andor SDK`
- The SDK path is configured in `HardwareManager` at `src/andor_qt/core/hardware_manager.py:77`
- **API reference**: `C:\Users\katsumilab\Downloads\Software Development Kit.pdf` (346 pages)
- **Hardware manual**: `C:\Users\katsumilab\Downloads\Andor_Newton_Manual.pdf` (44 pages)

### Required DLLs
- `atmcd64d.dll` — Camera control (CCD operations)
- `ShamrockCIF.dll` — Spectrograph control (gratings, wavelength)

### Mock Mode
- Set `ANDOR_MOCK=1` environment variable to force mock mode
- Mock implementations in `src/andor_pymeasure/instruments/mock.py`
- All tests run with `ANDOR_MOCK=1` automatically via `tests/conftest.py`

### SDK Return Codes
The Andor SDK uses integer return codes. Key ones:
- `20002` (DRV_SUCCESS) — Operation succeeded
- `20024` (DRV_TEMPERATURE_STABILIZED) — Cooler reached target
- `20034` (DRV_TEMPERATURE_NOT_REACHED) — Still cooling
- `20035` (DRV_TEMPERATURE_DRIFT) — Temperature drifting
- `20036` (DRV_TEMPERATURE_NOT_STABILIZED) — Not yet stabilized

### DU970P Newton EMCCD — Hardware Specifics

The camera is a **Newton DU970P EMCCD** (1600 × 200 pixels, 16 × 16 µm pixels, 16-bit).
It has **two output amplifiers** selectable in software:
- `type=0` → EM amplifier (use with EM gain for weak-signal / low-noise work)
- `type=1` → Conventional CCD amplifier (no EM multiplication, lower dark current)

#### VS Speeds (vertical shift, `SetVSSpeed(index)`)
| Index | Speed | Notes |
|-------|-------|-------|
| 0 | 4.9 µs/pixel | Fastest; slight CTE degradation possible |
| 1 | 9.8 µs/pixel | **Recommended default** (optimum CTE) |
| 2 | 19 µs/pixel | |
| 3 | 38 µs/pixel | |
| 4 | 57 µs/pixel | Slowest; best CTE |

Faster VS speeds increase clock-induced charge and heat at high frame rates (see Newton manual §6.2).

#### HS Speeds (horizontal readout, `SetHSSpeed(type, index)`)
| Index | Speed | Noise |
|-------|-------|-------|
| 0 | 3 MHz | Higher read noise (~tens of e⁻) |
| 1 | 1 MHz | Mid |
| 2 | 50 kHz | Lowest read noise |

Same speeds available for both EM and conventional amplifier.

#### EM Gain (`SetEMCCDGain(gain)`)
- Range: 1–1000 (queried at runtime via `GetEMGainRange()`)
- Only meaningful when EM amplifier (`type=0`) is selected
- Use ≥10 for EM noise benefit; at high gain EM noise factor = √2 (excess noise)
- Do NOT set EM gain when using conventional amplifier

#### Pre-Amplifier Gain (`SetPreAmpGain(index)`)
- Typically 3 options: ×1, ×2, ×4 (exact values queried at runtime via `GetNumberPreAmpGains / GetPreAmpGain`)
- Higher gain = better SNR for weak signals (lower effective read noise in e⁻/ADU)

#### Keep Cleans (`EnableKeepCleans(mode)`)
- `EnableKeepCleans(1)` — camera continuously flushes charge between external triggers to prevent dark current buildup
- `EnableKeepCleans(0)` — disable keep-clean cycles
- **Keep cleans OFF recommended for external trigger modes** — significantly reduces noise at high frame rates (500 Hz chopper_2x2)
- Keep cleans ON is useful for internal trigger with long idle periods between acquisitions
- Controlled via "Keep Cleans" checkbox in Camera Settings UI; default is OFF

#### Read Modes
- **FVB** (Full Vertical Binning): all 200 rows summed → 1D spectrum. Fastest for spectroscopy.
- **Image**: 2D readout with `SetImage(hbin, vbin, hstart, hend, vstart, vend)`
- **Single Track**: `SetSingleTrack(centre, height)` — bin `height` rows centred on `centre` (1-indexed from bottom)
- **Crop Mode**: `SetIsolatedCropMode(active, cropheight, cropwidth, vbin, hbin)`

#### Crop Mode — Critical Constraints (from SDK manual)
- Region is **always anchored to the bottom of the sensor** (nearest readout register). **Position CANNOT be adjusted on Newton cameras.**
- `SetIsolatedCropModeEx` (which allows position via `cropleft`/`cropbottom`) is **iXon Ultra only** — not available on DU970P.
- `SetCropMode` (simpler, FVB-only Newton variant) also anchors to bottom; only `cropHeight` parameter.
- **Light must NOT fall on excluded rows** — any charge there corrupts data through charge smear.
- `cropheight` and `cropwidth` must each be a multiple of their respective bin value.
- Newton supports FVB or Image read mode in isolated crop mode (iDus is FVB only).
- Achieves up to 1,515 spectra/sec with 20-row crop + 3 MHz HS speed.

#### `CameraSettingsWidget` — Usage Pattern
Shared widget in `src/andor_qt/widgets/hardware/camera_settings.py`.
Call `populate_from_camera(camera)` after hardware init to set runtime pre-amp options and EM gain range.
Returns settings via `get_settings()` dict → passed to `camera.apply_camera_settings(dict)`.
Used in: `DynamicInputsWidget`, `RealtimeWindow`, `TAScanConfigWidget`.

`get_settings()` keys: `trigger_mode`, `exposure_time` (seconds, float), `vs_speed`, `hs_speed`, `em_gain`, `pre_amp_gain`.
`exposure_spin` range: 1 µs–300 s (6 decimal places). Auto-set to 2 ms when `chopper_2x2` mode is selected in `TAScanConfigWidget`.

### Camera Shutdown
Always warm up before shutdown to prevent thermal damage:
```python
camera.set_cooler(False)  # Turn off cooler
# Wait for temp > -20°C before calling shutdown()
```
The GUI handles this automatically via `ShutdownDialog`.

## Motion Controllers

### Newport ESP302 — Delay Stage Controller

**Full API reference:** `ESP302-API-Reference.md` (in repo root)

#### Serial / TCP
- **RS-232**: 8N1, `rtscts=True` (**CRITICAL** — without it commands are silently dropped), 19200 baud typical
- **TCP port 5001**: raw ASCII (Telnet-style); **port 5002**: MKS Python wrapper library
- Commands end with `\r` (CR); responses end with `\r\n` (CR LF)

#### Command format (raw ASCII)
```
xxAAnnn\r      # xx = axis number (1–3), AA = mnemonic, nnn = params
1VA10\r        # set axis 1 velocity to 10 units/s
1TP\r          # query axis 1 position
```
Multiple commands in one line: semicolon-separated, max 80 chars.

#### Critical rules
| Rule | Detail |
|------|--------|
| **Motor on before move** | `xxMO` required; error `x13 MOTOR NOT ENABLED` if omitted |
| **No implicit wait** | ESP302 queues commands — always poll `xxMD?` (done=`1`) or send `xxWS` |
| **WS blocks serial** | In immediate mode `WS` freezes the port; use `MD?` polling from Python |
| **Home every power cycle** | `xxOR1` (switch + index); `MF` (soft off) retains position; `MK` clears it |
| **AB is all-axes** | `AB` stops and powers off ALL axes; no axis parameter |
| **PA over PR for precision** | Successive `PR` moves accumulate rounding error |

#### Key commands
| Command | Description |
|---------|-------------|
| `xxMO` / `xxMF` / `xxMK` | Motor ON / soft OFF / kill+clear-home |
| `xxOR1` | Home search (switch + index) |
| `xxPA{pos}` | Move to absolute position |
| `xxPR{delta}` | Move relative |
| `xxTP` | Read actual position |
| `xxMD?` | Motion done? `1`=done, `0`=moving |
| `xxVA{v}` / `xxAC{a}` | Set velocity / acceleration |
| `ST{ax}` / `AB` | Stop axis / emergency stop all |
| `TB` | Read oldest error (code, timestamp, message) |

#### Delay ↔ position conversion
```python
C_MM_PER_PS = 0.29979   # speed of light in mm/ps
def delay_ps_to_mm(delay_ps, reference_mm=0.0):
    return reference_mm + (delay_ps * C_MM_PER_PS) / 2   # /2 for double-pass
```

**Current project axis:** axis **2** (default changed from 1 to 2 in commit b5977be).

---

### OptoSigma GSC-02C + OSMS-YAW — Rotation Stage Controller

**Full API reference:** `GSC02-SGSP-YAW-Reference.md` (in repo root)

#### Serial connection
```python
serial.Serial(port='COM3', baudrate=9600, bytesize=8,
              parity='N', stopbits=1, rtscts=True, timeout=5)
```
- Factory DIP default: 9600 baud (SW2=OFF, SW1=ON)
- **`rtscts=True` required** — flow control is mandatory
- Commands end with `\r\n` (CR LF)

#### Critical rules
| Rule | Detail |
|------|--------|
| **M:/A:/J: need G:** | Motion commands only queue; **nothing moves until `G:` is sent** |
| **While BUSY** | Only `L:`, `Q:`, `!:`, `?:` accepted; all others silently ignored |
| **Homing direction** | OSMS-YAW: always `H:1-` (CW toward limit sensor) — `H:1+` runs CCW indefinitely |
| **No origin sensor** | OSMS-YAW has only a CW limit sensor; MINI homing touches it twice at two speeds |
| **`L:` won't stop `H:`** | Use `L:W` or `L:E` to abort a homing move |

#### Key commands
| Command | Returns | Description |
|---------|---------|-------------|
| `H:1-` | — | Home axis 1 CW (OSMS-YAW) |
| `M:1+P{n}` then `G:` | — | Relative move, n pulses CCW |
| `A:1+P{n}` then `G:` | — | Absolute move to pulse n (GSC-02C only) |
| `L:W` | — | Decelerate-stop all axes |
| `L:E` | — | Emergency stop all axes |
| `!:` | `B` or `R` | Quick busy check |
| `Q:` | `pos1,pos2,ACK1,ACK2,ACK3` | Full status (ACK3: `B`=busy, `R`=ready) |
| `R:W` | — | Zero coordinate at current position |

#### Stage specs (OSMS-YAW)
- 5-phase stepper, worm gear 1:144
- **400 pulses/degree** (half-step default, 0.0025°/pulse)
- CW hard stop at ~−2.5° (−1000 pulses); no CCW limit
- Positioning accuracy: 0.1°; repeatability: 0.02°; backlash: 0.1°

```python
PULSES_PER_DEGREE = 400
pulses = round(abs(degrees) * PULSES_PER_DEGREE)
```

#### DIP switch (OSMS-YAW required settings)
- SW3 = **OFF** (MINI homing — uses CW limit as reference)
- SW4 = **OFF** (Normal close limit sensor logic)

## TA (Transient Absorption) Module

### Overview

Pump-probe transient absorption measurement system. Key files:

| Class/File | Location | Purpose |
|-----------|----------|---------|
| `TAScanConfig` | `ta/scan_config.py` | Dataclass — all scan parameters |
| `TransientAbsorptionEngine` | `ta/engine.py` | Orchestrates scan loop in QThread |
| `TADataWriter` | `ta/hdf5_writer.py` | Writes ΔOD data to HDF5 files |
| `TAWindowPanel` | `windows/ta_panel.py` | Composite widget: config + live display |
| `TAScanConfigWidget` | `widgets/ta/scan_config_widget.py` | UI for configuring scans |
| `TALiveDisplayWidget` | `widgets/ta/live_display.py` | Real-time ΔOD plots |
| `AcquisitionSession` | `ta/acquisition.py` | Context manager: camera lifecycle across scan/monitor cycles |
| `NIDAQChopper500Hz` | `ta/nidaq_trigger.py` | Generates 500 Hz camera trigger via NI DAQ Counter 1 |
| `NIDAQPhaseReader` | `ta/nidaq_phase.py` | Reads chopper phase tags from P0.0 |

### Acquisition Modes

| Mode | Description |
|------|-------------|
| `boxcar` | Software alternation: first frame = pump-on, second = pump-off |
| `hardware` | NI DAQ hardware phase tags via P0.0; single shot per frame |
| `chopper_2x2` | 500 Hz camera trigger + chopper phase verification; 2 laser shots per frame |

### chopper_2x2 Mode — Hardware Wiring

**Concept:** 250 Hz chopper + 1 kHz laser → 500 Hz camera frames. Each frame integrates 2 laser shots (configurable via `shots_per_frame`). Camera is triggered externally by SDG at 500 Hz.

#### BNC-2110 Connections (current working setup)

| BNC-2110 Connector | NI PCIe-6353 Terminal | Direction | Signal |
|-------------------|----------------------|-----------|--------|
| User 1 BNC | PFI13 (P2.5, pin 40) | **INPUT** | Camera Fire output — phase reader start trigger |
| User 2 BNC | P0.0 (port0/line0) | **INPUT** | Chopper REF OUT (INNER) — phase tags |
| (dedicated BNC) | PFI0 | **INPUT** | 1 kHz laser sync clock |
| (dedicated BNC) | PFI12 (pin 2) | **INPUT** | SDG 500 Hz output (also chopper reference) |

Camera Ext Trigger SMB ← SDG 500 Hz (direct BNC connection, not through NI DAQ).

**CRITICAL — CTR0 conflict:** On PCIe-6353, CTR0 OUT is hardwired to PFI12 (pin 2) — the same pin used for the SDG input. **Never use CTR0 for counter output in this setup.**

#### Signal Chain

```
Laser 1 kHz → PFI0 (phase reader clock)
            → SDG (external reference)

SDG 500 Hz  → Camera Ext Trigger SMB (direct)
            → PFI12 (for NIDAQChopper500Hz retrigger, if used)
            → Chopper REF IN (chopper locks to SDG, runs at 250 Hz with ÷2)

Camera Fire output → PFI13 (User 1 BNC) — phase reader start trigger

Chopper REF OUT (INNER, 250 Hz) → P0.0 (User 2 BNC) — phase tags
```

#### Chopper Controller (Thorlabs MC2000B)

- **Blade:** MC1F10HP (10-slot high precision)
- **REF IN:** External, from SDG 500 Hz
- **Harmonic:** N=1, D=2 (divides SDG by 2 → 250 Hz chopper)
- **REF OUT:** INNER — tracks the actual blade position of inner slots
- **P0.0 = HIGH when blade is open** (beam passes through)

#### Chopper GUI Control (added 2026-05-23)

`andor_pymeasure.instruments.thorlabs_mc2000b` provides `MC2000B` (raw
serial) and `MockMC2000B` driver classes plus a `ChopperPair` helper that
wraps the pump-master / probe-slave pair. The pair is owned by
`HardwareManager.chopper_pair` and configured via `AppConfig.choppers`
(default pump=COM4, probe=COM3, blade=MC1F10HP, 250 Hz). `ChopperPair.shutdown()`
disables both choppers and closes their serial ports during shutdown.

`andor_qt.widgets.hardware.chopper_control.ChopperControlWidget` exposes
enable / phase / actual-frequency per chopper plus a "Setup pair" button
that applies the canonical configuration documented in
`project_chopper_working_config` memory. The widget is wired into
`TAWindowPanel` below the Scan/Monitor tabs; `camera_busy` from the TA
panel disables writes during a scan or monitor while keeping the
read-only Refresh button enabled.

Phase spinbox commits on Enter / focus-out / "Apply" click rather than
per-tick, so spinner clicks don't spam the serial line.

#### Camera Readout — Critical Lessons Learned

**Overlap mode is REQUIRED for 500 Hz with fast_external trigger.** Without overlap, the camera can only sustain ~260 Hz (exposure + readout > 2 ms frame period). With overlap enabled, readout happens during the next exposure, allowing 500 Hz. The camera may fire in bursts (not every trigger), but the frames that do fire are properly timed:
```python
self._sdk.SetOverlapMode(1)  # in start_run_till_abort()
```

**Without overlap, max frame rate is limited by exposure + readout:**
- VS=0, HS=0, hbin=1: readout ≈ 1.5 ms
- 0.4 ms exposure + 1.5 ms readout = 1.9 ms → max ~517 Hz
- 2.0 ms exposure + 1.5 ms readout = 3.5 ms → max ~285 Hz (too slow for 500 Hz)
- With overlap enabled, camera can sustain 500 Hz even with longer exposures

**Camera MUST NOT restart between acquisition cycles.** Restarting (`abort_acquisition` → `start_run_till_abort`) randomizes the phase relationship between camera frames and chopper blade, causing pump-ON/OFF assignment to flip randomly. The monitor engine now starts the camera once and keeps it running:
```python
# Monitor engine: camera starts ONCE before the loop
camera.start_run_till_abort()
phase_reader.start()
phase_reader.drain()
while not abort:
    frames = camera.get_buffered_frames()  # reads NEW frames only
    tags = phase_reader.read_tags(n * spf)
    delta = _process_chopper_frames(frames, tags, config)
camera.abort_acquisition()  # stops ONCE after the loop
```

`get_buffered_frames()` uses `GetNumberNewImages()` which returns only frames accumulated since the last read — safe for continuous operation. The circular buffer holds ~12000 frames (~24 seconds at 500 Hz).

**Phase sync on startup — Camera Fire trigger:** The phase reader uses the Camera Fire output (PFI13) as its NI DAQ start trigger. The reader arms but does not sample P0.0 until the camera actually begins its first exposure. This guarantees tag[0] corresponds to frame[0] deterministically. Tested 20/20 stable across camera restarts (vs 50/50 random without Fire trigger).

Previous approach (chopper REF on PFI13 as edge-detect sync) was confirmed NOT to work — the tag-to-frame alignment was still random because the camera and phase reader started on independent clocks.

**Tag-to-frame alignment:** The offset detection in `_process_chopper_frames` tries offsets 0 through `shots_per_frame-1` and picks the one with the most matched tag groups (all tags identical within a group). This works purely from the P0.0 tag pattern — it does NOT need intensity data or a chopped probe. With the Fire trigger, the offset is deterministic (always the same value across restarts).

**P0.0 polarity (MC2000B) — CAN flip between sessions.** This claim was wrong in earlier versions of this doc and was empirically corrected on 2026-05-20.

- **Within a single session (no chopper power-cycle, no PLL re-lock event), polarity is stable.** Validated by `scripts/test_fire_trigger.py`: 10/10 stable with Camera Fire trigger on PFI13 (vs 5/10 random without). The Fire trigger guarantees tag[0] = frame[0] deterministically.
- **Between sessions (after `enable=0`→`1`, power-cycle, or extended drift-and-re-lock), polarity CAN flip.** The MC2000B PLL at N/D=1/2 has TWO stable lock points 180° apart in the chopper cycle. Each re-lock event randomly picks one. The Camera Fire trigger keeps subsequent acquisitions consistent with whichever lock point the chopper happens to be in.
- **Mitigation:** `TAScanConfig.swap_tags: bool` (added 2026-05-20) and matching **"Swap pump-ON/OFF tags"** checkboxes in the TA Monitor and TA Scan panels. Toggle when the GUI shows pump-ON dark and pump-OFF bright. Persisted via QSettings.
- The photo-interrupter ↔ beam optical alignment offset is hardware-fixed, but which stable lock point the PLL chose determines the beam-to-interrupter phase relationship modulo 180° → which determines polarity.

Diagnostic: `scripts/test_tag_polarity_multiphase.py` runs the Fire-trigger acquisition at multiple chopper phases and reports STABLE ON-bright / STABLE OFF-bright / FLIPPED.

**`_ensure_idle()` prevents DRV_ACQUIRING (20072).** Called at the top of `start_run_till_abort()` and `start_run_till_abort_crop()` — checks `GetStatus()` and aborts if still acquiring.

#### NIDAQChopper500Hz (optional — when not using SDG directly)

Generates 500 Hz from 20 MHz timebase, retriggered by PFI12 (500 Hz SDG):

```python
trigger_gen = NIDAQChopper500Hz(
    device="Dev1",
    clock_source="/Dev1/PFI0",    # 1 kHz laser sync
    sync_source="/Dev1/PFI12",    # 500 Hz SDG → retrigger source
    counter="ctr1",               # CTR1 → PFI13 output
)
```

**Note:** nidaqmx 1.4.1 does not support `Toggle.PULSE` for divide-by-2. The counter uses the 20 MHz timebase with retriggerable start trigger on PFI12. Retrigger IS supported on PCIe-6353 (confirmed by diagnostic test). When using the SDG directly to trigger the camera, check "External trigger (SDG)" in the UI to skip starting this counter.

#### NIDAQPhaseReader (for chopper_2x2)

Reads `shots_per_frame` phase tags per camera frame from P0.0, clocked by PFI0 (one sample per laser shot):

```python
phase_reader = NIDAQPhaseReader(
    device="Dev1",
    di_channel="port0/line0",     # P0.0 = User 2 BNC = chopper REF OUT
    clock_source="/Dev1/PFI0",    # 1 kHz laser sync
    clock_rate=1000.0,
)
# shots_per_frame=2: tags [1,1]=pump-ON, [0,0]=pump-OFF, [1,0]=discard
# shots_per_frame=4: tags [1,1,1,1]=pump-ON, [0,0,0,0]=pump-OFF
```

#### chopper_2x2 Acquisition Logic

**`AcquisitionSession`** (context manager in `acquisition.py`) owns the camera lifecycle across multiple acquisitions. Both scan and monitor engines use it:

```python
with AcquisitionSession(hw, config, camera_settings, phase_reader) as session:
    for delay_ps in delays:
        axis.position_ps = delay_ps
        delta = session.acquire_one_cycle(dark=dark, raw_callback=cb)
```

- `__enter__`: applies camera settings, starts camera (`start_run_till_abort`), starts phase reader, drains tags
- `acquire_one_cycle()`: sleeps for frame accumulation, reads buffered frames + tags, calls `_process_chopper_frames()`
- `__exit__`: stops camera (`abort_acquisition`)
- For non-chopper modes (`boxcar`, `shot_to_shot`): delegates to existing mode-specific functions

**`_process_chopper_frames()`** is the single source of truth for tag alignment and frame separation. It tries offsets 0 through `shots_per_frame-1` and picks the offset with the most matched groups (all tags in a group identical).

Frame assignment: P0.0=1 → pump-ON, P0.0=0 → pump-OFF by default. If `TAScanConfig.swap_tags=True`, the assignment is inverted (P0.0=0 → pump-ON). **No intensity-based auto-swap** — trust the chopper controller output, then let the user override via the swap-tags checkbox if the PLL lock point puts the polarity backward.

#### Camera Settings for chopper_2x2

- **Trigger mode:** `fast_external` — camera waits for external trigger on Ext Trigger SMB
- **Exposure time:** 0.4 ms (must be ≤ 0.5 ms for 500 Hz without overlap)
- **VS speed:** index 0 (4.9 µs/row) — fastest, required for high frame rates
- **HS speed:** index 0 (3 MHz) — fastest
- **Overlap:** ENABLED (required for 500 Hz frame rate — DO NOT DISABLE)
- **shots_per_frame:** 2 (500 Hz camera / 250 Hz chopper) or 4 (250 Hz camera / 125 Hz chopper)
- **CRITICAL: `apply_camera_settings` key names must match exactly:**
  - `vs_speed_index` (not `vs_speed`) — without this, VS speed is never set and readout is too slow
  - `hs_speed_index` (not `hs_speed`) — same issue
  - `amplifier_type`, `preamp_gain_index`, `em_gain` — all use `_index` suffix
  - Wrong key names are silently ignored, causing subtle timing failures

### TAScanConfig — NI DAQ Fields

| Field | Default | Description |
|-------|---------|-------------|
| `nidaq_device` | `"Dev1"` | NI DAQ device name |
| `nidaq_di_channel` | `"port0/line0"` | P0.0 — digital input for phase reader |
| `nidaq_clock_source` | `"/Dev1/PFI0"` | 1 kHz laser sync (phase reader sample clock) |
| `nidaq_clock_rate` | `1000.0` | Clock rate in Hz |
| `nidaq_chopper_sync_source` | `"/Astrella DAQ/PFI12"` | 250 Hz chopper pulse (counter retrigger source) |
| `nidaq_chopper_counter` | `"ctr1"` | NI counter to use for 500 Hz trigger generation |
| `nidaq_fire_trigger` | `"/Astrella_DAQ/PFI13"` | Camera Fire output terminal — phase reader start trigger. Guarantees tag[0] = frame[0]. |
| `swap_tags` | `False` | Invert P0.0 polarity interpretation (tag=0 → pump-ON instead of tag=1). UI exposed as "Swap pump-ON/OFF tags" checkbox in Monitor + TA Scan panels. |

### Chopper Diagnostic Scripts (scripts/)

Standalone scripts under `scripts/` for debugging the chopper_2x2 setup. None of them depend on the andor_qt GUI; they share defaults matching the Monitor configuration (exposure=0.4 ms, vs=0, hs=0, amp=1, preamp=0, hbin=8, fast_external) and expose every camera setting as a CLI flag.

| Script | Purpose |
|--------|---------|
| `chopper_control.py` | Raw MC2000B serial control via pyserial — bypasses Thorlabs DLL (which times out on v4.19 firmware). Subcommands: `list`, `info`, `setup`, `phase`, `enable`, `sweep`, `shutdown`. |
| `acquire_dark_frame.py` | Records dark spectrum (probe blocked at source) for dark-subtracted contrast. Outputs .npy. |
| `chopper_phase_sweep.py` | Full COM4 phase sweep with contrast measurement at each point. Optional `--dark-frame`, `--lock-timeout`, `--samples-per-phase`. Restarts camera per phase (bulletproof atexit + signal shutdown). |
| `test_chopper_lock_dynamics.py` | Polls `locked?` at 10 Hz for 20 s at each test phase. Reports first-lock time, lock fraction, streak durations. No camera needed. Use to choose dwell + lock-timeout. |
| `test_tag_polarity_multiphase.py` | Fire-trigger acquisition at multiple phases, N trials each. Reports STABLE ON-bright / OFF-bright / FLIPPED. Use to confirm tag stability + identify which phases give which polarity. |
| `test_fire_trigger.py` | A/B compares phase reader WITHOUT Camera Fire trigger (5/10 random flips) vs WITH (10/10 stable). Confirms Fire trigger is wired correctly. |
| `model_chopper_phase.py` | Pure-Python simulation of 250 Hz square chopper + 1 kHz laser deltas + 500 Hz camera with 0.4 ms exposure. Validates expectations against measured data. |
| `analyze_sweep_precision.py` | From a sweep CSV, extracts σ-per-pair and predicts σ at any (pairs N, samples K). Empirical v4.19: σ_per_pair ≈ 13.4 % contrast → 200 pairs gives σ ≈ 0.95 %. |

**MC2000B firmware v4.19 fact**: rejects any fractional phase like `phase=19.5` with `CMD_ARG_INVALID`. Only integer phases 0-360 accepted. The phase sweep script rounds internally before sending.

**MC2000B SDK DLL fact**: `MC2000CommandLibWin32.dll` (and Win64) timed out (-3) against firmware v4.19. The Thorlabs Python SDK is unusable on this firmware. **Use raw pyserial via the chopper_control.py module — the ASCII protocol from the device manual chapter 8 works fine at 115200 8N1.**

### Mock Objects (ANDOR_MOCK=1)

| Class | Location | Pattern |
|-------|----------|---------|
| `MockNIDAQChopper500Hz` | `ta/nidaq_trigger.py` | No hardware; has `is_running` property |
| `MockNIDAQChopper2x2Reader` | `ta/nidaq_phase.py` | Returns `[1,1],[0,0],[1,1]…` — matched pairs |
| `MockNIDAQPhaseReader` | `ta/nidaq_phase.py` | Returns alternating `1,0,1,0…` — single-shot pattern |

`_make_chopper_2x2_hardware()` in `ta_panel.py` selects mock vs real objects based on `ANDOR_MOCK` env var.

---

## Architecture

### Singleton Pattern
- `HardwareManager` — Single instance managing camera/spectrograph
- `EventBus` — Pub/sub for inter-widget communication
- Access via `.instance()` class method

### Signal Flow
```
Hardware → HardwareSignals → Widgets
                ↓
            EventBus → Cross-widget updates
```

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `HardwareManager` | `core/hardware_manager.py` | Singleton managing hardware lifecycle |
| `SpectrumPlotWidget` | `widgets/display/spectrum_plot.py` | Multi-trace spectrum plot |
| `TraceListWidget` | `widgets/display/trace_list.py` | Trace visibility management |
| `ExperimentQueueRunner` | `core/experiment_queue.py` | Sequential procedure execution |
| `SequencerAdapter` | `core/sequencer_adapter.py` | PyMeasure SequencerWidget integration |
| `DynamicInputsWidget` | `widgets/inputs/dynamic_inputs.py` | Parameter input form |
| `RealtimeWindow` | `windows/realtime_window.py` | Continuous live acquisition window |
| `DataSettingsWidget` | `widgets/hardware/data_settings.py` | Save directory and metadata settings |
| `TransientAbsorptionEngine` | `ta/engine.py` | TA scan loop in QThread with pause/abort |
| `TAWindowPanel` | `windows/ta_panel.py` | Composite TA panel; wires engine + NI DAQ hardware |
| `NIDAQChopper500Hz` | `ta/nidaq_trigger.py` | 500 Hz camera trigger via NI DAQ Counter 1 |
| `NIDAQPhaseReader` | `ta/nidaq_phase.py` | Chopper phase tags from P0.0 per laser shot |

### Procedures
PyMeasure procedures in `src/andor_qt/procedures/`:
- `SpectrumProcedure` — FVB spectrum acquisition
- `ImageProcedure` — 2D image acquisition

Both use `SharedHardwareMixin` from `procedures/base.py` to share hardware with GUI.

## Testing

### Philosophy: Functional Tests + Smoke Tests

Tests should be layered. **Smoke tests** catch regressions (does it run
without crashing?). **Functional tests** verify correctness (does it
produce the right output?). Both are needed — but every component should
have at least one functional test that asserts computed values.

**Smoke test — catches regressions:**
```python
def test_process_chopper_frames_no_crash(self):
    """Basic smoke test — function runs without exception."""
    result = _process_chopper_frames(frames, tags, config)
    assert result is not None
    assert result.shape == (n_pixels,)
```

**Functional test — verifies correctness:**
```python
def test_process_chopper_frames_correct_delta(self):
    """ON=1200, OFF=1000 → ΔI/I₀ = (1200-1000)/1000 = 0.2"""
    frames = np.array([[1200]*5, [1000]*5, [1200]*5, [1000]*5], dtype=float)
    tags = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    result = _process_chopper_frames(frames, tags, config)
    np.testing.assert_allclose(result, 0.2, atol=1e-10)
```

**Rule: never write only smoke tests.** A smoke test is fine as a starting
point, but always follow it with a functional test that checks the actual
output against a hand-calculated expected value.

### What To Test For Each Function

| Function type | Assert | Example |
|---------------|--------|---------|
| **Math/computation** | Exact numerical output matches hand-calculated expected value | `assert_allclose(delta, 0.2)` |
| **Data pipeline** | Output shape, dtype, and values match known input→output mapping | `assert result.shape == (200,)` |
| **Signal emission** | Signal payload contains correct data (not just that it fired) | `assert payload[0] == expected_wavelengths` |
| **File I/O** | Written file contents match expected format and values | Read file back, check each field |
| **State machine** | State transitions produce expected side effects | After pause→resume, verify scan continues |
| **Error paths** | Correct exception type AND message for invalid input | `pytest.raises(RuntimeError, match="no valid pairs")` |
| **Edge cases** | Boundary values, empty inputs, division by zero | `_process_chopper_frames(empty_frames, ...)` raises |

### Test Structure
```
tests/
├── conftest.py          # Global fixtures, mock SDK setup
├── ta/
│   ├── test_process_chopper_frames.py  # Core frame processing math
│   ├── test_acquisition_session.py     # AcquisitionSession lifecycle + incremental
│   ├── test_acquisition_hardware.py    # acquire_delta_signal_at_delay all modes
│   ├── test_engine.py                  # Scan engine signals, status, chopper, static
│   ├── test_engine_helpers.py          # Pure helpers: _format_eta, _make_scan_folder
│   ├── test_monitor_engine.py          # Monitor continuous, static, single-phase
│   ├── test_hdf5_writer.py            # HDF5 I/O, metadata, CSV export
│   └── test_scan_config.py            # Delay generators, unit conversions
├── qt/
│   ├── conftest.py      # Qt fixtures (qt_app, hardware_manager, wait_for)
│   ├── test_ta_integration.py         # TA panel wiring, hbin wavelength alignment, chopper widget
│   ├── test_hardware_manager.py       # Singleton, init/shutdown, chopper_pair lifecycle
│   └── widgets/
│       ├── test_camera_settings_widget.py  # Camera UI, QSettings persistence
│       ├── test_chopper_control_widget.py  # MC2000B pair widget: refresh, setup, busy state
│       └── test_ta_live_display.py         # Display, hover, click-to-select, readout
├── instruments/
│   └── test_thorlabs_mc2000b.py       # Chopper driver (Mock + serial FakeSerial)
├── integration/         # Hardware tests (--hardware flag)
├── procedures/          # Procedure tests
└── e2e/                 # End-to-end workflow tests
```

### Writing New Tests — Checklist

For each function under test:
- [ ] **Happy path**: known input → verify exact expected output
- [ ] **Edge case**: empty input, zero values, boundary values
- [ ] **Error case**: invalid input → correct exception type and message
- [ ] **Side effects**: signals emitted with correct payload, stats updated, files written
- [ ] **Integration**: component A feeds component B → verify end-to-end data flow

### Hardware Integration Tests

Tests under `tests/integration/` require real hardware. They are marked with
`@pytest.mark.hardware` and **skipped by default**. Run with:
```bash
uv run pytest tests/integration/ --hardware -v
```

### Hardware Fixture Lifecycle Rules

**All hardware fixtures MUST be function-scoped** (not session or module).
The Andor camera SDK locks the DLL exclusively — if a fixture keeps the
camera initialized after the test, subsequent tests (or the GUI) cannot
acquire it.

Rules for hardware integration test fixtures:
- **Function scope** — `@pytest.fixture` (no `scope=` argument). Init in setup, shutdown in teardown.
- **Always teardown** — Use `yield` + cleanup after yield.
- **Abort before shutdown** — If the test starts acquisition, the `finally` block must call `camera.abort_acquisition()`.
- **No leftover NI DAQ tasks** — Use `with nidaqmx.Task() as task:` context managers.
- **Pop ANDOR_MOCK** — Hardware fixtures must `os.environ.pop("ANDOR_MOCK", None)`.

### Key Fixtures (Mock Tests)
- `qt_app` — QApplication instance (module scope)
- `hardware_manager` — Fresh HardwareManager with mock SDK
- `wait_for` — Polling helper for async operations

### Qt Signal Testing
Signals from background threads need event loop processing:
```python
def wait_for_qt(condition_fn, timeout=15.0):
    app = QApplication.instance()
    start = time.time()
    while time.time() - start < timeout:
        if app:
            app.processEvents()
        if condition_fn():
            return True
        time.sleep(0.05)
    return False
```

### CI

GitHub Actions runs all non-hardware tests on push/PR to master.
See `.github/workflows/test.yml`. A post-edit ruff hook in
`.claude/settings.json` auto-fixes lint on every Python file edit.

## Common Patterns

### Adding a New Widget
1. Create widget in appropriate `widgets/` subdirectory
2. Export from `__init__.py`
3. Add to `main_window.py` layout
4. Connect signals in `_connect_signals()`
5. Write tests in `tests/qt/`

### Adding a New Procedure
1. Create procedure class inheriting from `Procedure` and `SharedHardwareMixin`
2. Define parameters as class attributes using `Parameter()`
3. Implement `startup()`, `execute()`, `shutdown()`
4. Inject hardware via `HardwareManager.inject_into_procedure()`

### Thread Safety
- Use Qt signals for cross-thread communication
- `AcquisitionSignals` class in main_window provides thread-safe signals
- Never access Qt widgets from background threads

## Recent Features (2024)

### Spectrum Overlay (Commits 17-19)
- `SpectrumPlotWidget.add_trace()` — Add overlay trace
- `TraceListWidget` — Manage trace visibility
- Up to 20 traces with automatic color cycling

### PyMeasure Sequencer (Commits 20-22)
- `ExperimentQueueRunner` — Background queue execution
- `SequencerAdapter` — Bridges SequencerWidget to our queue
- QTabWidget with Single/Sequence tabs in left panel

## Recent Features (2026)

### Swap pump-ON/OFF tags (May 2026)
- `TAScanConfig.swap_tags: bool` — invert P0.0 polarity interpretation in `_process_chopper_frames`. Honors PLL bistability: chopper REF OUT polarity can flip between sessions; this flag corrects it without touching hardware.
- UI: checkbox below "External trigger" in `TAMonitorWidget` and `TAScanConfigWidget`. Persisted via QSettings.
- Disabled while a scan/monitor is running.

### Default exposure time persistence (May 2026)
- `CameraSettingsWidget` initial exposure is **0.4 ms** (was 2 ms).
- Persistence bug fixed: `_on_acq_mode_changed` runs `apply_mode_preset` on widget construction which clobbered QSettings; both `TAMonitorWidget` and `TAScanConfigWidget` now re-call `_camera_settings._load_settings()` after `_build_ui` finishes, so the persisted value wins.
- `apply_mode_preset()` still hard-codes 0.002/0.0003 for chopper_2x2/shot_to_shot — these only kick in when the user actively switches mode in the dropdown.

### Bulletproof script camera shutdown (May 2026)
Scripts that touch the Andor SDK (`scripts/chopper_phase_sweep.py`, `scripts/acquire_dark_frame.py`) register an `atexit` handler at module load that always calls `camera.shutdown()` if the camera was initialized. Also install signal handlers for SIGINT / SIGBREAK / SIGTERM that raise KeyboardInterrupt so try/finally unwinds normally. Pattern to copy when writing new SDK-touching scripts:

```python
_camera_for_cleanup = None

def _emergency_shutdown():
    global _camera_for_cleanup
    cam = _camera_for_cleanup
    if cam is None:
        return
    _camera_for_cleanup = None
    try: cam.abort_acquisition()
    except Exception: pass
    try: cam.shutdown()
    except Exception as e: print(f"[atexit] {e}", flush=True)

atexit.register(_emergency_shutdown)

def _signal_handler(signum, _frame):
    raise KeyboardInterrupt()
for n in ("SIGINT", "SIGBREAK", "SIGTERM"):
    s = getattr(signal, n, None)
    if s is not None:
        try: signal.signal(s, _signal_handler)
        except (ValueError, OSError): pass

# In init_camera():
_camera_for_cleanup = cam  # register BEFORE settings apply

# In finally:
_camera_for_cleanup = None  # already shut down — prevent atexit double-call
```

The only way to leak the SDK after these handlers is a hard `taskkill /F` — which we never do (per the project rule).

## Instrument Driver Patterns

### Creating New Instruments

Follow PyMeasure's Instrument pattern:
```python
from pymeasure.instruments import Instrument

class MyInstrument(Instrument):
    """Docstring describing the instrument."""

    # Use property creators instead of getter/setter methods
    voltage = Instrument.control(
        "VOLT?", "VOLT %g",
        """Control output voltage in V (float from 0 to 10).""",
        validator=strict_range,
        values=[0, 10],
    )

    # Read-only properties use measurement()
    status = Instrument.measurement(
        "STATUS?",
        """Measure the device status (str, read-only).""",
    )

    # Write-only properties use setting()
    reset = Instrument.setting(
        "*RST",
        """Set to True to reset the device.""",
    )
```

### Validators

- `strict_range` — Raises ValueError if out of range
- `truncated_range` — Clamps value to range silently
- `strict_discrete_set` — Raises if not in allowed set
- `truncated_discrete_set` — Rounds to nearest allowed value

### Mock Instruments

For testing without hardware:
- Inherit from base instrument class
- Override adapter with mock responses
- Use `ANDOR_MOCK=1` environment variable
- Mock implementations in `src/andor_pymeasure/instruments/`

## Non-Obvious Conventions

### Signal Parameter Order
When signals pass multiple objects, order matters:
```python
# Correct order for spectrum signals
spectrum_ready = Signal(object, object, dict)  # wavelengths, intensities, params
image_ready = Signal(object, object, dict)     # image, wavelengths, params
```

### Hardware Manager Singleton
Always use `.instance()`, never construct directly:
```python
# Correct
hw = HardwareManager.instance()

# Wrong - creates orphan instance
hw = HardwareManager()
```

### Qt Thread Safety
Never access widgets from background threads. Use signals:
```python
# In background thread - emit signal
self._acq_signals.spectrum_ready.emit(wavelengths, data, params)

# In main thread - slot handles update
@Slot(object, object, dict)
def _on_spectrum_ready(self, wavelengths, data, params):
    self._plot.add_trace(wavelengths, data)  # Safe - main thread
```

### Procedure Parameter Injection
Procedures get shared hardware via mixin, not constructor:
```python
class SpectrumProcedure(SharedHardwareMixin, Procedure):
    def startup(self):
        self._init_hardware()  # Gets shared instances from HardwareManager
```

## Troubleshooting

### "Hardware not initialized"
- Check SDK path in `hardware_manager.py`
- Verify DLLs are present
- Try mock mode: `--mock`

### Tests timeout waiting for signals
- Use `wait_for_qt()` instead of `wait_for()` for Qt signals
- Ensure `QApplication.processEvents()` is called in wait loop

### Camera won't cool
- Check cooler is enabled: `camera.set_cooler(True)`
- Verify temperature target: `camera.set_temperature(-60)`
- Monitor via `temperature_changed` signal

### TA Monitor / Scan shows pump-ON dark and pump-OFF bright (inverted polarity)
- Expected behavior — MC2000B PLL has 2 stable lock points (at N/D=1/2), polarity may flip between sessions.
- Fix: check the **"Swap pump-ON/OFF tags"** box in the Monitor or TA Scan panel.
- Diagnose: run `scripts/test_tag_polarity_multiphase.py --phases 19,20,21 --trials 5` — confirms whether polarity is stable (in which case the swap-tags toggle is the right fix) or randomly flipping mid-acquisition (which would point to a Fire-trigger problem).

### Chopper SDK calls time out
- Don't use Thorlabs' `MC2000CommandLibWin*.dll` against firmware v4.19 — it times out (-3) on every Get* call.
- Use the raw pyserial path via `scripts/chopper_control.py` instead. ASCII protocol at 115200 8N1 works fine.

### Chopper rejects fractional phase like `phase=19.5`
- MC2000B firmware v4.19 only accepts integer phases. Returns `CMD_ARG_INVALID` for any decimal. Use `int(round(phase))` before sending.
