# VQC v0 架構與執行結果

`vqc_v0.py` 是一個使用 Qiskit 與 PyTorch 建立的最小可行變分量子電路（Variational Quantum Circuit, VQC）範例。它將參數化量子電路包裝成 PyTorch 模型，並對一個合成二元分類任務進行訓練。

## 1. 模型規格

- 量子位元數：4
- 輸入特徵數：4
- 資料編碼：每個 qubit 套用一個 `RY(x[i])`
- 可訓練層數：2
- 每層可訓練門：`RY(theta)` 與 `RZ(theta)`
- 糾纏方式：linear CX chain
- 可訓練參數總數：16
- 輸出 observable：qubit 0 上的 Pauli-Z 期望值
- 優化器：Adam，learning rate 為 0.1
- 損失函數：Binary Cross-Entropy Loss
- 訓練回合：30 epochs

## 2. 程式架構

### 2.1 套件與隨機種子

程式使用 NumPy 產生資料與初始參數，使用 PyTorch 管理模型、損失函數和優化流程，再使用 Qiskit 建立參數化量子電路。

NumPy 與 PyTorch 的隨機種子都設為 `0`，使每次執行時的資料、權重初始值與訓練結果可重現。

### 2.2 超參數

```python
N_QUBITS = 4
N_LAYERS = 2
ENTANGLE_MODE = "linear"
```

`ENTANGLE_MODE` 可設為 `linear` 或 `circular`。Circular 模式會在每層多加一個從最後一個 qubit 到第一個 qubit 的 CX 門。

### 2.3 VQC 電路建立

`build_vqc_circuit()` 會建立兩組參數：

- `x_params`：代表輸入資料，共 4 個參數。
- `theta_params`：代表模型權重，共 `2 layers × 4 qubits × 2 rotations = 16` 個參數。

電路先使用 `RY(x[i])` 將輸入特徵編碼到量子狀態，然後在每個可訓練層中套用 `RY` 與 `RZ` 旋轉，最後以 CX 門建立 qubits 之間的糾纏。

### 2.4 PyTorch 模型包裝

`VQCModel` 繼承 `torch.nn.Module`，並透過 `TorchConnector` 將 `EstimatorQNN` 轉換為可參與 PyTorch 自動微分的量子 layer。

量子電路輸出的 Pauli-Z 期望值介於 `-1` 與 `1` 之間，因此 forward pass 會將它轉換到 `0` 與 `1` 之間：

```python
return (raw + 1) / 2
```

轉換後的數值可以直接作為二元分類機率使用。

### 2.5 Observable

```python
observable = SparsePauliOp.from_list([("IIIZ", 1)])
```

Qiskit 的 Pauli 字串使用 little-endian qubit ordering，因此 `IIIZ` 代表量測 qubit 0 的 Pauli-Z 期望值。

### 2.6 合成資料集

程式產生 40 筆四維資料，每個特徵從 `[-pi, pi]` 均勻取樣。標籤規則為：

```text
sum(x) > 0  -> label 1
sum(x) <= 0 -> label 0
```

這是一個簡單的線性可分任務，用來驗證 VQC 與 PyTorch 的整合、前向運算和反向傳播都能正常運作。

### 2.7 訓練流程

每個 epoch 的執行順序為：

1. 清除上一輪的梯度。
2. 將 40 筆訓練資料傳入 VQC。
3. 計算 Binary Cross-Entropy Loss。
4. 透過 `loss.backward()` 計算量子電路參數的梯度。
5. 使用 Adam 更新 `theta` 參數。
6. 每 5 個 epochs 顯示 loss 與 accuracy。

## 3. 量子電路圖

![VQC v0 quantum circuit](vqc_v0_circuit.png)

電路由左至右的順序為：

1. `RY(x[0...3])` 資料編碼。
2. 第一層 `RY(theta) + RZ(theta)` 可訓練旋轉。
3. 第一組 `q0 -> q1 -> q2 -> q3` linear CX 糾纏。
4. 第二層 `RY(theta) + RZ(theta)` 可訓練旋轉。
5. 第二組 linear CX 糾纏。
6. 量測 qubit 0 的 Pauli-Z 期望值作為模型輸出。

## 4. 實際執行輸出

在專案的 `qiskit_env` 環境中執行 30 epochs 後，得到以下結果：

```text
Epoch  0 | Loss: 0.6617 | Acc: 0.62
Epoch  5 | Loss: 0.5284 | Acc: 0.75
Epoch 10 | Loss: 0.5252 | Acc: 0.80
Epoch 15 | Loss: 0.5334 | Acc: 0.80
Epoch 20 | Loss: 0.5289 | Acc: 0.82
Epoch 25 | Loss: 0.5222 | Acc: 0.80
Epoch 29 | Loss: 0.5208 | Acc: 0.77
```

## 5. 結果解讀

- Loss 從 `0.6617` 下降到 `0.5208`，表示模型確實學到了部分分類規則。
- Accuracy 從 `62%` 提升，最高在 epoch 20 達到 `82%`。
- 最後 accuracy 為 `77%`，顯示訓練後期仍有一些波動。
- 波動可能與資料僅有 40 筆、沒有獨立驗證集，以及 learning rate `0.1` 偏高有關。
- 目前的 accuracy 是訓練集 accuracy，不能視為模型對未見資料的泛化能力。

## 6. 執行方式

在 repository 目錄中使用已安裝 Qiskit Machine Learning 與 PyTorch 的 Python 環境執行：

```bash
python vqc_v0.py
```

Qiskit 可能會顯示自動建立 gradient function 的訊息。這是提示訊息，不是執行錯誤。
