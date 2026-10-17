# 參與開發

歡迎修正錯誤或補上功能，送 pull request 前請先跑過下面的檢查。

---

## 開發環境

需要 Python 3.10 以上。建議使用虛擬環境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 安裝本專案

```bash
pip install -e ".[all,dev]"
radar-stream-decoder --help
```

`all` 是 pyserial、PySide6、matplotlib 與 pandas。只裝 `.[dev]` 也能跑測試：GUI 與 DataFrame 的測試會自動略過，序列埠的測試用模擬埠，不需要 pyserial。

---

## 提交前的檢查

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

CI 會執行相同的檢查，在 Linux 與 Windows 上跑 Python 3.10–3.13，另有一個不裝選用套件的工作，以及一個釘在最舊相依版本（numpy 1.24、typer 0.12、pyserial 3.5、pandas 1.5、pytest 7.2）的工作。GUI 測試以 `QT_QPA_PLATFORM=offscreen` 執行，不需要螢幕。

### Python 3.10 相容性

本專案宣告 `requires-python = ">=3.10"`，請避開 3.11 以後才有的語法與標準函式庫：

| 不要用 | 請用 |
| --- | --- |
| `tomllib` | 設定檔是 JSON，不需要 TOML |
| `typing.Self` | 類別名稱字串 |
| `enum.StrEnum` | `class Foo(str, Enum)` |
| `datetime.UTC` | `datetime.timezone.utc` |
| `except*`、`ExceptionGroup` | 一般的 `except` 與單一例外 |
| `asyncio.TaskGroup` | `asyncio.gather` |
| PEP 695（`class Foo[T]`、`type X = ...`） | `TypeVar` 與一般的型別註記 |
| `itertools.batched` | 明確的切片迴圈 |
| `np.float_`、`np.NaN`、`np.trapz` 等 NumPy 2.0 移除或改名的名稱 | `np.float64`、`np.nan`、明確的加總 |

任何 Python 版本都能執行的 3.10 語法閘門（與 CI 相同，略過 `.venv`、`build`、`dist`）：

```bash
python -c "import ast,pathlib; [ast.parse(p.read_text(encoding='utf-8'), str(p), feature_version=(3, 10)) for p in pathlib.Path('.').rglob('*.py') if not {'.venv', 'build', 'dist', '__pycache__'} & set(p.parts)]"
```

`tests/test_python310_compat.py` 會自動掃描上述語法、標準函式庫與 NumPy 名稱，並確認匯入套件時不會載入選用套件。

---

## 程式碼風格

- 每個模組開頭加 `from __future__ import annotations`。
- 公開函式要有 type hints 與英文 numpy 風格 docstring；實作文獻方法時，在 `References` 段標明出處。
- 封包欄位沿用 TI 文件與韌體原始碼的名稱（`numDetectedObj`、`xyzQFormat`、`rangeIdx`、`dopplerIdx`……），註解寫明依據的是哪個 SDK 或 Toolbox 版本的哪個檔案。文件與韌體不一致時以韌體為準，並在註解說明差在哪裡。
- 用 `logging` 輸出診斷訊息，不要用 `print`；CLI 的結果摘要用 `typer.echo`。
- 陣列運算盡量向量化，不寫逐元素的迴圈。
- 設定放在 `DecoderConfig`／`ViewerConfig`，不要在程式裡寫死路徑或數值。
- README 有英文（`README.md`）與繁體中文（`README.zh-TW.md`）兩份，改一份時另一份要在同一個 commit 一起改；CHANGELOG 與本檔案用繁體中文。

---

## 資料

**本 repository 不包含任何錄製資料。** 測試只使用程式合成的封包，以及依 TI 韌體 C 結構以 `struct.pack` 打包的參考封包。請不要提交擷取檔（`.bin`、`.dat`）、解碼輸出（`.csv`、`.npz`、`.jsonl`、`report.json`）或任何含個人資訊的檔案。`.gitignore` 已經忽略這些副檔名以及 `captures/`、`outputs/` 兩個資料夾，提交前仍請用 `git status` 確認一次。`configs/` 裡的 `.cfg` 是範例組態，會進版本庫。

---

## 提交訊息

提交訊息用英文：第一行寫清楚改了什麼，需要時空一行，再用幾句話說明原因；小修正可以用 `fix: ...` 開頭。有功能變動時，請同步更新 `CHANGELOG.md` 的 `[Unreleased]` 段。
