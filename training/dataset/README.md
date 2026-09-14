# 最終740特徴モデルの学習データ

コンペ参加時の`PTCG-kaggle-tetsuro731`リポジトリにある
`data/catboost/lucario_v3_combined/datasets/20260802-14-top20-v8-window/`
から、2026-08-02〜14の13 shardを全量コピーしました。
上位プレイヤーの勝利episode由来の、最終モデルと同じv8・740特徴量です。
全1,558 episode、90,029判断、688,908候補行を含みます。

元データは非圧縮NPZで約2 GBありましたが、`numpy.savez_compressed`で約35.8 MB
（34.1 MiB）になりました。各配列の値・dtypeは元ファイルと全要素照合済みです。
episodeや特徴量の間引き、数値の丸めは行っていません。通常のGit cloneで取得でき、LFSは不要です。

- `rows_*.npz`: 日付ごとの候補特徴量・教師ラベル・判断メタデータ。
- `features.json`: コピー元と同一の特徴量名・順序・バージョン。
- `dataset_manifest.json`: 出所、件数、各shardの容量とSHA-256。
- `episode_ids.json`: 全1,558 episode ID（昇順、重複なし）。
- `benchmark.json`: 全量・最終条件での再学習時間、環境、パラメータ、training指標。

NPZの`X`は候補ごとの特徴量、`y`は教師が選んだ候補の0/1ラベル、
`group`は候補から判断への対応です。`dec_sizes`は判断ごとの候補数、
`dec_episode`はshard内のepisode番号、`dec_episode_id`は元のepisode IDです。
その他の`dec_*`配列は選択数・コンテキスト・ルールベース一致判定などの判断単位の情報です。
`features.json`の`names`が`X`の列順に対応します。
既存の`load_shards`がshardを連結するときにgroupとepisodeの内部番号を補正します。

学習コマンドは[ルートREADME](../../README.md#training-from-included-data)を参照してください。
`--final-fit`では全量を学習し、指標も同じ学習データ上で計算します。
検証付きの学習を試す場合は`--final-fit`を外すと、seedに基づいてepisode単位で
20%を検証へ分け、early stoppingを適用します。ただし特徴量schemaと最終パラメータは
コンペ時に選定済みであり、新たな独立評価を保証する分割ではありません。
