# PTCG AI Battle Challenge - Silver Medal Solution

Kaggle Simulation Competition「Pokemon TCG AI Battle Challenge」で使用した、
Mega Lucario exデッキの最終提出コードです。最終結果は銀メダルでした。

## Approach

ルールベース方策を土台に、上位プレイヤーの勝利episodeから学習したCatBoostの
行動スコアラーを組み合わせました。モデルは
`MAIN`、`TO_HAND`、`DISCARD`、`ATTACH_FROM`、`SWITCH`の選択だけに使用し、対象外の場面や
推論失敗時にはルールベース方策へフォールバックします。

最終CatBoostモデルは740特徴量、3,373 treesです。提出環境では外部パッケージを
必要としないよう、学習済みモデルを純Pythonの木走査コードへ変換しています。

## Validation

- Training AUC: 0.97567
- 1件選択の一致率: 81.69%（ルールベース: 57.72%）
- 学習データ: 1,558 episode、90,029判断、688,908候補行

## Files

- `docs/WRITEUP_JA.md`: 最終解法レポート（日本語）
- `docs/WRITEUP_EN.md`: 最終解法レポート（英語）
- `docs/images/`: 解法レポートの画像とSVG生成元
- `submission/main.py`: エージェント本体とルールベース方策
- `submission/features_window.py`: 最終740次元の特徴量生成
- `submission/features_*.py`: 最終特徴量の構成モジュール
- `submission/damage.py`, `submission/incoming_damage.py`: ダメージ関連特徴量
- `submission/model_data.py`: 純Pythonへ変換したCatBoostモデル
- `submission/deck.csv`: 最終提出の60枚デッキ
- `artifacts/model_manifest.json`: 学習条件と評価指標
- `training/dataset/`: 最終740特徴モデルの変換済み全量データ（1,558 episode、約35.8 MB）
- `training/config_window.json`: 最終740特徴モデルの学習設定
- `training/`: 旧514特徴モデルの学習設定とepisode分割
- `scripts/catboost/`: データ生成、学習、評価、純Pythonモデルへの変換

## Training from Included Data

最終提出モデルに使用した変換済みデータを全量同梱しています。
replayのダウンロード・変換やCompetition SDKなしで学習できます。
Python 3.11と`uv`を使用します。

```bash
uv sync --locked
uv run python -m scripts.catboost.train \
    --config training/config_window.json \
    --data training/dataset \
    --iterations 3373 --depth 9 --lr 0.05 \
    --l2-leaf-reg 3.748639774100797 \
    --random-strength 0.379062406483889 \
    --positive-class-weight 2.0963135011151595 \
    --subsample 0.7668472842334632 --rsm 0.966539533174889 \
    --seed 7 --threads 16 --final-fit
```

全1,558 episode・90,029判断・688,908候補行・740特徴量で、
最終提出時と同じパラメータを使って3,373 treesを学習します。
`--final-fit`は検証集合を設けず全量を学習するため、表示される指標は**学習データ上の値**です。
対戦勝率や未知データへの精度ではありません。

Intel Core i7-12700K（CPU 16スレッド）、Python 3.11.15、CatBoost 1.2.10で、
**全量・最終条件の学習は340.36秒（約5.7分）**でした。
データ読込・モデル保存・指標計算などを含む、結果JSON保存直前までの実測は347.48秒です。
依存関係のインストール時間は含まず、時間は実行環境によって変わります。
Training AUC 0.97567、1件選択の一致率81.69%で、既存の最終モデルmanifestにある
全指標（context別を含む）と一致しました。実測記録は
[`benchmark.json`](training/dataset/benchmark.json)に保存しています。

モデル・指標・特徴量重要度・学習時間は、表示される
`data/catboost/lucario_v3_window/runs/<run-id>/`へ保存されます。
データの出所・形式・実測結果は[データセットの説明](training/dataset/README.md)を参照してください。

## Earlier Training Workflow

以下は旧514特徴モデル用の手順です。Python 3.11と`uv`を使用します。Competition SDKの`cg`は
`data/extracted/sample_submission/cg`へ配置してください。

```bash
uv sync
uv run python -m scripts.catboost.generate_data \
	--config training/config.json \
	--from 2026-08-02 --to 2026-08-08 \
	--keep-episodes training/episode_ids.json \
	--out data/catboost/lucario_v3

uv run python -m scripts.catboost.evaluate \
	--data data/catboost/lucario_v3

uv run python -m scripts.catboost.train \
	--config training/config.json \
	--data data/catboost/lucario_v3 \
	--iterations 3000 --depth 6 --lr 0.05 --seed 7 \
	--validation-episodes training/holdout_episode_ids.json
```

学習後に表示されるrun directoryを指定すると、CatBoostモデルを提出用の
純Pythonコードへ変換できます。

```bash
uv run python -m scripts.catboost.export \
	--config training/config.json \
	--run data/catboost/lucario_v3/runs/<run-id> \
	--out generated_model_data.py
```

Competition SDK、対戦replay、CatBoostの`.cbm`モデルは含みません。
最終740特徴モデルの全量データとepisode IDは`training/dataset/`に同梱しています。
旧514特徴モデルのデータ生成には、別途SDKとreplayが必要です。
