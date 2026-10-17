# 變更紀錄

本檔案記錄 `radar-stream-decoder` 的所有重要變更。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/)。

## [Unreleased]

---

## [0.1.0] - 2026-10-17

### 新增

- 首次釋出。
- 可任意切塊餵入、遇錯自動重新同步的 TLV 組包器，支援 SDK 1.x／2.x／3.x out-of-box demo 與 Vital Signs lab 的輸出格式。格式依 header 的形狀判別（offset 12 的欄位、checksum、TLV 長度加總），不只看 SDK 版本字；判斷不出來的封包以 `unsupported-format` 回報，不硬套格式。SDK 3.x 的 padding 是韌體未初始化的緩衝區，只記一筆 `uneven-padding` 資訊，截斷與否看長度與下一個同步字。
- 三種追蹤韌體的格式：`people-counting-3d`（mmWave Industrial Toolbox 4.6–4.12 的 3D People Counting，48 位元組含 checksum 的 header、壓縮球座標點雲、112／40 位元組目標、每點目標編號、presence）、`people-counting-2d`（Industrial Toolbox 2.x 的 2D people counting，52 位元組含 checksum 與時間戳記的 header、浮點極座標點雲、68 位元組目標、每點目標編號）、`people-tracking`（Radar Toolbox 的 people tracking，TLV 1020／1010／1011／1012／1021）。包含資料模型、自動判別與 `--schema` 覆寫、checksum 驗證、球座標轉直角座標、合成資料與錯誤注入、`inspect` 追蹤統計，以及檢視器上的目標標記。2D people counting 的時間戳記依韌體（`task_mbox.c` 在 R4F 上蓋時間戳記）以 200 MHz 換算。
- 每點目標編號依 TI 的定義掛回上一幀的點（`tracking.link_target_associations()`）；點雲 CSV 帶有 `snr_linear` 與 `target_id` 欄，`decode` 另寫出目標清單 `targets.csv`，`stream` 可用 `--targets-csv` 邊收邊寫。
- 由雷達 `.cfg` 推導 FMCW 參數，點雲、距離剖面與熱圖以物理單位輸出；`sensor-config` 可選擇三項慣例（`--wavelength-at start/center`、`--max-range-factor`、`--doppler-padding/--no-doppler-padding`），輸出以 `convention_*` 標明使用的慣例，預設為 TI mmWave Demo Visualizer 的慣例。都卜勒 bin 索引一律以 FFT bin 間距（`SensorConfig.doppler_bin_mps`）換算速度。輸出包括 `center_frequency_ghz`、`if_range_limit_m`、`doppler_bin_mps`，另附 IWR6843 三發四收的範例組態 `iwr6843_3d_3tx4rx`。
- log2 幅度轉 dB 依韌體合併天線的方式（`DecoderConfig.log2_combining`：`average`、`sum`、`shifted-sum`）；SDK 3.x 依 TI mmWave SDK 03.05.00.04 韌體的算法，Q8、每根虛擬天線先右移 ⌈log₂ N⌉ 再累加。
- 擷取檔鑑識：去除重複、frame 遺失、感測器重啟、frame 週期、鏈路負載與追蹤統計；沒有時間戳記的 48 位元組 header 以 frame 編號與 frame 週期計時。
- 序列埠即時串流（`[serial]`）與 PySide6 即時檢視器（`[gui]`）；`viz --schema` 同時決定合成示範的韌體格式，資訊面板顯示追蹤目標數。CLI 埠的交握依 TI mmWave SDK 03.05.00.04 的 CLI 原始碼：送組態前清空輸入緩衝區，`Error <碼>` 與不認得的指令立刻回報並指出行號，能處理不換行的提示字元與被逾時切開的行，`configDataPort` 改了資料埠 baud 時主機端跟著換。序列埠開不了、讀取失敗或感測器拒絕指令時，CLI 只印一行訊息（感測器的多行回應以分號接成一行）並以結束碼 2 結束，traceback 只在 DEBUG 記錄。
- 依同一批原始碼寫成的模擬 CLI 埠與資料埠（`port_emulator`：開機 banner、不換行的提示字元、回顯、`Done`／`Error <碼>`／`Skipped`、未知指令、`Ignored` 訊息、校正狀態訊息、`configDataPort` 換 baud 與 16 個 `0xFF`），沒有硬體也能跑完交握與背景讀取。
- 命令列介面 `radar-stream-decoder` 與 Python API；匯出 CSV、NPZ、JSON Lines 與 pandas DataFrame。
- 以合成封包與錯誤注入為基礎的測試套件，不依賴任何實機擷取資料；另有依 TI 韌體 C 結構以 `struct.pack` 打包、期望值依韌體演算法獨立算出的參考測試（`tests/test_firmware_reference.py`、`tests/test_people_counting.py`）。
- `examples/` 的五支範例，全部只用合成資料。
- 解碼設定集中在 `DecoderConfig`，可由 JSON 設定檔載入（`--config`）；擷取檔與輸出路徑一律由命令列或 API 參數指定。
- 英文版 README（`README.md`，GitHub 預設顯示），繁體中文版在 `README.zh-TW.md`；兩份都有支援格式與驗證程度總表、TI 文件與韌體不一致之處。
- 流程圖分英文與繁體中文兩版：英文版是 `docs/workflow.html` 與 `docs/images/workflow.png`，繁體中文版是 `docs/workflow.zh-TW.html` 與 `docs/images/workflow.zh-TW.png`。
